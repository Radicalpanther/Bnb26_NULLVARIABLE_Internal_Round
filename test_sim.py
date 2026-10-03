#!/usr/bin/env python3
"""
Full simulation test script.
"""

import json
import random
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

# Knowledge Base with ground truth and plausible wrong alternatives
COMPANIES = [
    {"name": "Apple", "ceo": "Tim Cook", "founded": 1976, "hq_city": "Cupertino", "country": "USA", "valuation_b": 3000, "product": "iPhone", "wrong_ceo": "Andy Jassy", "wrong_hq": "Mountain View", "wrong_val": 1800, "wrong_prod": "Vision Pro"},
    {"name": "Microsoft", "ceo": "Satya Nadella", "founded": 1975, "hq_city": "Redmond", "country": "USA", "valuation_b": 2800, "product": "Azure", "wrong_ceo": "Sundar Pichai", "wrong_hq": "Seattle", "wrong_val": 1200, "wrong_prod": "Surface"},
    {"name": "NVIDIA", "ceo": "Jensen Huang", "founded": 1993, "hq_city": "Santa Clara", "country": "USA", "valuation_b": 2200, "product": "GPUs", "wrong_ceo": "Mark Zuckerberg", "wrong_hq": "San Jose", "wrong_val": 650, "wrong_prod": "Drive Thor"},
    {"name": "Alphabet", "ceo": "Sundar Pichai", "founded": 1998, "hq_city": "Mountain View", "country": "USA", "valuation_b": 1800, "product": "Google Search", "wrong_ceo": "Tim Cook", "wrong_hq": "Cupertino", "wrong_val": 2800, "wrong_prod": "Gemini"},
    {"name": "Amazon", "ceo": "Andy Jassy", "founded": 1994, "hq_city": "Seattle", "country": "USA", "valuation_b": 1700, "product": "AWS", "wrong_ceo": "Satya Nadella", "wrong_hq": "Denver", "wrong_val": 240, "wrong_prod": "Prime Video"},
    {"name": "Meta", "ceo": "Mark Zuckerberg", "founded": 2004, "hq_city": "Menlo Park", "country": "USA", "valuation_b": 1200, "product": "Instagram", "wrong_ceo": "Jensen Huang", "wrong_hq": "San Francisco", "wrong_val": 1700, "wrong_prod": "Quest Pro"},
    {"name": "Tesla", "ceo": "Elon Musk", "founded": 2003, "hq_city": "Austin", "country": "USA", "valuation_b": 650, "product": "Model Y", "wrong_ceo": "Patrick Collison", "wrong_hq": "Austin", "wrong_val": 380, "wrong_prod": "Cybertruck"},
    {"name": "ASML", "ceo": "Christophe Fouquet", "founded": 1984, "hq_city": "Veldhoven", "country": "Netherlands", "valuation_b": 380, "product": "EUV Lithography", "wrong_ceo": "Daniel Ek", "wrong_hq": "Munich", "wrong_val": 70, "wrong_prod": "DUV Tools"},
    {"name": "TSMC", "ceo": "C.C. Wei", "founded": 1987, "hq_city": "Hsinchu", "country": "Taiwan", "valuation_b": 750, "product": "Semiconductors", "wrong_ceo": "Alex Karp", "wrong_hq": "Tokyo", "wrong_val": 270, "wrong_prod": "Foundry Nodes"},
    {"name": "Spotify", "ceo": "Daniel Ek", "founded": 2006, "hq_city": "Stockholm", "country": "Sweden", "valuation_b": 65, "product": "Music Streaming", "wrong_ceo": "Shantanu Narayen", "wrong_hq": "Zurich", "wrong_val": 43, "wrong_prod": "Podcasts Plus"},
]

CITIES = [
    {"city": "Cupertino", "country": "USA", "population": 60000, "elevation_m": 72, "timezone": "PST", "wrong_pop": 82000, "wrong_elev": 32},
    {"city": "Redmond", "country": "USA", "population": 76000, "elevation_m": 13, "timezone": "PST", "wrong_pop": 737000, "wrong_elev": 53},
    {"city": "Santa Clara", "country": "USA", "population": 127000, "elevation_m": 22, "timezone": "PST", "wrong_pop": 1013000, "wrong_elev": 25},
    {"city": "Mountain View", "country": "USA", "population": 82000, "elevation_m": 32, "timezone": "PST", "wrong_pop": 60000, "wrong_elev": 72},
    {"city": "Seattle", "country": "USA", "population": 737000, "elevation_m": 53, "timezone": "PST", "wrong_pop": 76000, "wrong_elev": 13},
    {"city": "Menlo Park", "country": "USA", "population": 34000, "elevation_m": 22, "timezone": "PST", "wrong_pop": 127000, "wrong_elev": 22},
    {"city": "Austin", "country": "USA", "population": 974000, "elevation_m": 149, "timezone": "CST", "wrong_pop": 873000, "wrong_elev": 16},
    {"city": "Veldhoven", "country": "Netherlands", "population": 45000, "elevation_m": 22, "timezone": "CET", "wrong_pop": 145000, "wrong_elev": 12},
    {"city": "Hsinchu", "country": "Taiwan", "population": 450000, "elevation_m": 30, "timezone": "CST", "wrong_pop": 975000, "wrong_elev": 28},
    {"city": "Stockholm", "country": "Sweden", "population": 975000, "elevation_m": 28, "timezone": "CET", "wrong_pop": 450000, "wrong_elev": 30},
]

SCIENTISTS = [
    {"name": "Marie Curie", "birth_city": "Warsaw", "birth_year": 1867, "field": "Physics and Chemistry", "discovery": "Radium and Polonium", "wrong_field": "Mathematics", "wrong_disc": "General Relativity"},
    {"name": "Alan Turing", "birth_city": "London", "birth_year": 1912, "field": "Computer Science", "discovery": "Turing Machine and Enigma Decryption", "wrong_field": "Quantum Electrodynamics", "wrong_disc": "Normal Distribution"},
    {"name": "Albert Einstein", "birth_city": "Ulm", "birth_year": 1879, "field": "Theoretical Physics", "discovery": "General Relativity", "wrong_field": "Electrical Engineering", "wrong_disc": "Alternating Current System"},
    {"name": "Nikola Tesla", "birth_city": "Smiljan", "birth_year": 1856, "field": "Electrical Engineering", "discovery": "Alternating Current System", "wrong_field": "Biophysics", "wrong_disc": "DNA Photo 51"},
    {"name": "Ada Lovelace", "birth_city": "London", "birth_year": 1815, "field": "Mathematics", "discovery": "First Computer Algorithm", "wrong_field": "Theoretical Physics", "wrong_disc": "General Relativity"},
    {"name": "Carl Friedrich Gauss", "birth_city": "Braunschweig", "birth_year": 1777, "field": "Mathematics", "discovery": "Normal Distribution and Number Theory", "wrong_field": "Computer Science", "wrong_disc": "Turing Machine"},
]

ELEMENTS = [
    {"element": "Titanium", "symbol": "Ti", "atomic_number": 22, "melting_point_c": 1668, "density_g_cm3": 4.506, "wrong_an": 29, "wrong_mp": 1085, "wrong_den": 8.96},
    {"element": "Platinum", "symbol": "Pt", "atomic_number": 78, "melting_point_c": 1768, "density_g_cm3": 21.45, "wrong_an": 79, "wrong_mp": 1064, "wrong_den": 19.32},
    {"element": "Gold", "symbol": "Au", "atomic_number": 79, "melting_point_c": 1064, "density_g_cm3": 19.32, "wrong_an": 47, "wrong_mp": 961.8, "wrong_den": 10.49},
    {"element": "Copper", "symbol": "Cu", "atomic_number": 29, "melting_point_c": 1085, "density_g_cm3": 8.96, "wrong_an": 22, "wrong_mp": 1668, "wrong_den": 4.506},
    {"element": "Silicon", "symbol": "Si", "atomic_number": 14, "melting_point_c": 1414, "density_g_cm3": 2.329, "wrong_an": 92, "wrong_mp": 1132, "wrong_den": 19.1},
    {"element": "Silver", "symbol": "Ag", "atomic_number": 47, "melting_point_c": 961.8, "density_g_cm3": 10.49, "wrong_an": 78, "wrong_mp": 1768, "wrong_den": 21.45},
]

# Uniform Paraphrased Templates
PLAN_INPUT_TEMPLATES = [
    "User Goal: {prompt}",
    "Objective: {prompt}",
    "Task: {prompt}",
    "Request: {prompt}",
    "Query: {prompt}"
]

PLAN_OUTPUT_TEMPLATES = [
    "1. {s1}\n2. {s2}\n3. {s3}",
    "Plan steps:\n- {s1}\n- {s2}\n- {s3}",
    "Execution outline: First, {s1}. Next, {s2}. Finally, {s3}.",
    "Step 1: {s1}. Step 2: {s2}. Step 3: {s3}."
]

SELECT_TOOL_INPUT_TEMPLATES = [
    "Select tool for sub-goal: '{query}'",
    "Determine appropriate tool for query: '{query}'",
    "Choose tool to execute: '{query}'",
    "Select optimal tool for: '{query}'",
    "Identify tool for operation: '{query}'"
]

SELECT_TOOL_OUTPUT_TEMPLATES = [
    "Selected tool: '{tool}'. Strategy: Route execution to {tool}.",
    "Tool choice: '{tool}'. Reasoning: Best fit for query requirements.",
    "Selected tool '{tool}' to handle the requested operation.",
    "Using '{tool}' to fulfill sub-goal.",
    "Proceeding with '{tool}' for this step."
]

CALL_TOOL_INPUT_TEMPLATES = [
    "Invoke {tool} with query: '{query}'",
    "Calling {tool} with input: '{query}'",
    "Execute {tool} query: '{query}'",
    "Run {tool} on: '{query}'",
    "Send request to {tool}: '{query}'"
]

CALL_TOOL_OUTPUT_SEARCH_TEMPLATES = [
    "[Search Result] {content}",
    "Search output: {content}",
    "Retrieved document: {content}",
    "[Result from Search] {content}"
]

CALL_TOOL_OUTPUT_CALC_TEMPLATES = [
    "[Calculator Output] {val}",
    "Result: {val}",
    "Calculated value: {val}",
    "Output: {val}"
]

CALL_TOOL_OUTPUT_DB_TEMPLATES = [
    "{json_data}",
    "[Database Record] {json_data}",
    "DB Result: {json_data}"
]

READ_RESULT_INPUT_TEMPLATES = [
    "Parse and extract key facts from step {dep_idx}",
    "Read tool output from step {dep_idx}",
    "Extract relevant information from step {dep_idx}",
    "Process output received at step {dep_idx}",
    "Parse result from step {dep_idx}"
]

READ_RESULT_OUTPUT_TEMPLATES = [
    "Extracted value: '{val}'. Successfully parsed.",
    "Extracted: '{val}'. Verified and stored in state.",
    "Parsed value '{val}' from tool output.",
    "Retrieved value: '{val}'.",
    "Stored '{val}' into working memory."
]

WRITE_ANSWER_INPUT_TEMPLATES = [
    "Synthesize final answer for goal: '{prompt}'",
    "Generate response to user query: '{prompt}'",
    "Formulate final response based on retrieved facts",
    "Write answer to task objective: '{prompt}'",
    "Synthesize answer from state for: '{prompt}'"
]

WRITE_ANSWER_OUTPUT_TEMPLATES = [
    "Answer: {ans}",
    "Based on the retrieved data, {ans}",
    "{ans}",
    "Final Answer: {ans}"
]

@dataclass
class StepRecord:
    run_id: str
    step_index: int
    step_type: str
    tool_name: str
    input_text: str
    output_text: str
    latency_ms: float
    error_flag: int
    retry_count: int
    output_length: int
    state_snapshot: str
    depends_on: str
    is_root_cause: int

@dataclass
class RunRecord:
    run_id: str
    task_type: str
    final_outcome: str
    fault_type: str
    root_cause_step_index: Any
    total_steps: int

ALL_TOOLS = ["search", "calculator", "database_lookup"]
FAULT_TYPES = [
    "wrong_tool_selected",
    "bad_tool_arguments",
    "tool_error_ignored",
    "context_ignored",
    "corrupted_state",
    "retry_loop"
]

@dataclass
class TaskSpec:
    task_type: str
    prompt: str
    ground_truth_answer: str
    plausible_wrong_answer: str
    plan_steps: Tuple[str, str, str]
    hops: List[Dict[str, Any]]

def generate_task_spec(task_type: str, rng: random.Random) -> TaskSpec:
    if task_type == "fact_lookup":
        template_choice = rng.choice(["company_ceo", "company_founded", "scientist_discovery", "element_props", "city_stats"])

        if template_choice == "company_ceo":
            comp = rng.choice(COMPANIES)
            prompt = f"Who is the current CEO of {comp['name']} and what is their primary product?"
            gt = f"The CEO of {comp['name']} is {comp['ceo']}, and their primary product is {comp['product']}."
            wrong_ans = f"The CEO of {comp['name']} is {comp['wrong_ceo']}, and their primary product is {comp['wrong_prod']}."
            plan = (f"Search knowledge base for {comp['name']} executive leadership", "Extract CEO name and flagship product", "Return factual answer")
            hops = [{
                "tool_name": "search",
                "query": f"{comp['name']} CEO and primary product",
                "wrong_query": f"{comp['name']} history archive",
                "raw_result": f"{comp['name']} leadership overview: The CEO is {comp['ceo']}. The company is famous for its {comp['product']}.",
                "wrong_raw_result": f"{comp['name']} leadership overview: The CEO is {comp['wrong_ceo']}. The company is famous for its {comp['wrong_prod']}.",
                "extracted_val": f"{comp['ceo']} (CEO), {comp['product']} (Product)",
                "wrong_extracted_val": f"{comp['wrong_ceo']} (CEO), {comp['wrong_prod']} (Product)",
                "extracted_dict": {"ceo": comp["ceo"], "product": comp["product"]},
                "wrong_extracted_dict": {"ceo": comp["wrong_ceo"], "product": comp["wrong_prod"]}
            }]

        elif template_choice == "company_founded":
            comp = rng.choice(COMPANIES)
            prompt = f"In what year was {comp['name']} founded, and where is its headquarters located?"
            gt = f"{comp['name']} was founded in {comp['founded']} and is headquartered in {comp['hq_city']}, {comp['country']}."
            wrong_ans = f"{comp['name']} was founded in {comp['founded'] + 5} and is headquartered in {comp['wrong_hq']}, {comp['country']}."
            plan = (f"Lookup {comp['name']} founding details in database", "Extract founding year and HQ location", "Format answer")
            hops = [{
                "tool_name": "database_lookup",
                "query": f"SELECT founded_year, hq_city, country FROM companies WHERE name = '{comp['name']}'",
                "wrong_query": f"SELECT founded_year, hq_city FROM companies WHERE id = 0",
                "raw_result": json.dumps({"company": comp["name"], "founded_year": comp["founded"], "hq_city": comp["hq_city"], "country": comp["country"]}),
                "wrong_raw_result": json.dumps({"company": comp["name"], "founded_year": comp["founded"] + 5, "hq_city": comp["wrong_hq"], "country": comp["country"]}),
                "extracted_val": f"{comp['founded']} (Founded), {comp['hq_city']}, {comp['country']} (HQ)",
                "wrong_extracted_val": f"{comp['founded'] + 5} (Founded), {comp['wrong_hq']}, {comp['country']} (HQ)",
                "extracted_dict": {"founded": comp["founded"], "hq_city": comp["hq_city"]},
                "wrong_extracted_dict": {"founded": comp["founded"] + 5, "hq_city": comp["wrong_hq"]}
            }]

        elif template_choice == "scientist_discovery":
            sci = rng.choice(SCIENTISTS)
            prompt = f"What field of study was {sci['name']} known for, and what was their major discovery?"
            gt = f"{sci['name']} worked in {sci['field']} and is famous for {sci['discovery']}."
            wrong_ans = f"{sci['name']} worked in {sci['wrong_field']} and is famous for {sci['wrong_disc']}."
            plan = (f"Query biographical repository for {sci['name']}", "Extract academic field and breakthrough", "Write response")
            hops = [{
                "tool_name": "search",
                "query": f"Biographical profile of {sci['name']} field and discovery",
                "wrong_query": f"General archive of {sci['name']} publications",
                "raw_result": f"{sci['name']} (born {sci['birth_year']} in {sci['birth_city']}): Renowned for work in {sci['field']}. Key breakthrough: {sci['discovery']}.",
                "wrong_raw_result": f"{sci['name']} (born {sci['birth_year']} in {sci['birth_city']}): Renowned for work in {sci['wrong_field']}. Key breakthrough: {sci['wrong_disc']}.",
                "extracted_val": f"Field: {sci['field']}; Discovery: {sci['discovery']}",
                "wrong_extracted_val": f"Field: {sci['wrong_field']}; Discovery: {sci['wrong_disc']}",
                "extracted_dict": {"field": sci["field"], "discovery": sci["discovery"]},
                "wrong_extracted_dict": {"field": sci["wrong_field"], "discovery": sci["wrong_disc"]}
            }]

        elif template_choice == "element_props":
            elem = rng.choice(ELEMENTS)
            prompt = f"What is the atomic number and melting point of {elem['element']} in Celsius?"
            gt = f"{elem['element']} has an atomic number of {elem['atomic_number']} and a melting point of {elem['melting_point_c']} °C."
            wrong_ans = f"{elem['element']} has an atomic number of {elem['wrong_an']} and a melting point of {elem['wrong_mp']} °C."
            plan = (f"Lookup physical constants of {elem['element']}", "Extract atomic number and melting point", "Output verified values")
            hops = [{
                "tool_name": "database_lookup",
                "query": f"SELECT atomic_number, melting_point_c FROM elements WHERE element = '{elem['element']}'",
                "wrong_query": f"SELECT atomic_number FROM elements WHERE symbol = 'X'",
                "raw_result": json.dumps({"element": elem["element"], "atomic_number": elem["atomic_number"], "melting_point_c": elem["melting_point_c"]}),
                "wrong_raw_result": json.dumps({"element": elem["element"], "atomic_number": elem["wrong_an"], "melting_point_c": elem["wrong_mp"]}),
                "extracted_val": f"Atomic #{elem['atomic_number']}, Melting Point: {elem['melting_point_c']} °C",
                "wrong_extracted_val": f"Atomic #{elem['wrong_an']}, Melting Point: {elem['wrong_mp']} °C",
                "extracted_dict": {"atomic_number": elem["atomic_number"], "melting_point_c": elem["melting_point_c"]},
                "wrong_extracted_dict": {"atomic_number": elem["wrong_an"], "melting_point_c": elem["wrong_mp"]}
            }]

        else:  # city_stats
            city = rng.choice(CITIES)
            prompt = f"What is the population and elevation of {city['city']}, {city['country']}?"
            gt = f"{city['city']} has a population of {city['population']:,} residents and an elevation of {city['elevation_m']} meters."
            wrong_ans = f"{city['city']} has a population of {city['wrong_pop']:,} residents and an elevation of {city['wrong_elev']} meters."
            plan = (f"Query geographic database for {city['city']}", "Extract population count and elevation", "Format answer")
            hops = [{
                "tool_name": "database_lookup",
                "query": f"SELECT population, elevation_m FROM cities WHERE city = '{city['city']}'",
                "wrong_query": f"SELECT population FROM cities WHERE city = 'UNKNOWN'",
                "raw_result": json.dumps({"city": city["city"], "population": city["population"], "elevation_m": city["elevation_m"]}),
                "wrong_raw_result": json.dumps({"city": city["city"], "population": city["wrong_pop"], "elevation_m": city["wrong_elev"]}),
                "extracted_val": f"Population: {city['population']:,}, Elevation: {city['elevation_m']}m",
                "wrong_extracted_val": f"Population: {city['wrong_pop']:,}, Elevation: {city['wrong_elev']}m",
                "extracted_dict": {"population": city["population"], "elevation_m": city["elevation_m"]},
                "wrong_extracted_dict": {"population": city["wrong_pop"], "elevation_m": city["wrong_elev"]}
            }]

    elif task_type == "math_lookup":
        template_choice = rng.choice(["compound_interest", "discount_tax", "profit_margin", "cylinder_volume"])

        if template_choice == "compound_interest":
            p = rng.choice([5000, 10000, 15000, 20000, 25000])
            r = rng.choice([4.5, 5.0, 6.0, 7.5])
            t = rng.choice([2, 3, 4, 5])
            val = round(p * ((1 + r / 100.0) ** t), 2)
            wrong_val = round(p * (1 + (r / 100.0) * t), 2)  # Simple interest instead of compound
            expr = f"{p} * (1 + {r}/100)**{t}"
            wrong_expr = f"{p} * (1 + ({r}/100) * {t})"
            prompt = f"Calculate the compound interest future value for a principal of ${p:,} at an annual interest rate of {r}% over {t} years."
            gt = f"The future value after {t} years is ${val:,.2f}."
            wrong_ans = f"The future value after {t} years is ${wrong_val:,.2f}."
            plan = ("Formulate compound interest expression", "Call calculator tool", "Write final result")
            hops = [{
                "tool_name": "calculator",
                "query": expr,
                "wrong_query": wrong_expr,
                "raw_result": str(val),
                "wrong_raw_result": str(wrong_val),
                "extracted_val": f"${val:,.2f}",
                "wrong_extracted_val": f"${wrong_val:,.2f}",
                "extracted_dict": {"future_value": val},
                "wrong_extracted_dict": {"future_value": wrong_val}
            }]

        elif template_choice == "discount_tax":
            price = rng.choice([120, 250, 450, 800, 1200])
            disc = rng.choice([10, 15, 20, 25])
            tax = rng.choice([5, 8, 10])
            val = round(price * (1 - disc / 100.0) * (1 + tax / 100.0), 2)
            wrong_val = round(price * (1 - (disc - 5) / 100.0) * (1 + tax / 100.0), 2)
            expr = f"{price} * (1 - {disc}/100) * (1 + {tax}/100)"
            wrong_expr = f"{price} * (1 - {disc}/100)"
            prompt = f"An item with an initial price of ${price} has a {disc}% discount applied, followed by {tax}% sales tax. What is the final price?"
            gt = f"The final price after a {disc}% discount and {tax}% tax is ${val:,.2f}."
            wrong_ans = f"The final price after a {disc}% discount and {tax}% tax is ${wrong_val:,.2f}."
            plan = ("Compute discounted base price", "Apply sales tax", "Present final price")
            hops = [{
                "tool_name": "calculator",
                "query": expr,
                "wrong_query": wrong_expr,
                "raw_result": str(val),
                "wrong_raw_result": str(wrong_val),
                "extracted_val": f"${val:,.2f}",
                "wrong_extracted_val": f"${wrong_val:,.2f}",
                "extracted_dict": {"final_price": val},
                "wrong_extracted_dict": {"final_price": wrong_val}
            }]

        elif template_choice == "profit_margin":
            rev = rng.choice([500000, 750000, 1200000, 2400000])
            cogs = rng.choice([200000, 320000, 480000, 900000])
            opex = rng.choice([80000, 120000, 250000, 400000])
            net = rev - cogs - opex
            margin = round((net / rev) * 100.0, 2)
            wrong_margin = round(((rev - cogs) / rev) * 100.0, 2)  # Gross margin instead of net
            expr = f"(({rev} - {cogs} - {opex}) / {rev}) * 100"
            wrong_expr = f"(({rev} - {cogs}) / {rev}) * 100"
            prompt = f"A company reported revenue of ${rev:,}, COGS of ${cogs:,}, and operating expenses of ${opex:,}. What is the net profit margin percentage?"
            gt = f"The net profit margin is {margin:.2f}% (Net income: ${net:,})."
            wrong_ans = f"The net profit margin is {wrong_margin:.2f}% (Gross income: ${rev - cogs:,})."
            plan = ("Compute net profit (Revenue - COGS - OpEx)", "Calculate margin percentage", "Write answer")
            hops = [{
                "tool_name": "calculator",
                "query": expr,
                "wrong_query": wrong_expr,
                "raw_result": str(margin),
                "wrong_raw_result": str(wrong_margin),
                "extracted_val": f"{margin:.2f}%",
                "wrong_extracted_val": f"{wrong_margin:.2f}%",
                "extracted_dict": {"net_margin": margin},
                "wrong_extracted_dict": {"net_margin": wrong_margin}
            }]

        else:  # cylinder_volume
            radius = rng.choice([3, 4, 5, 6, 8])
            height = rng.choice([7, 10, 12, 15])
            vol = round(3.14159265 * (radius ** 2) * height, 2)
            wrong_vol = round(3.14159265 * radius * (height ** 2), 2)
            expr = f"3.14159265 * ({radius}**2) * {height}"
            wrong_expr = f"3.14159265 * {radius} * ({height}**2)"
            prompt = f"Calculate the volume of a cylinder with radius r = {radius} cm and height h = {height} cm (in cubic centimeters)."
            gt = f"The volume of the cylinder is {vol:,.2f} cm³."
            wrong_ans = f"The volume of the cylinder is {wrong_vol:,.2f} cm³."
            plan = ("Apply formula V = π * r^2 * h", "Compute result with calculator", "Return volume")
            hops = [{
                "tool_name": "calculator",
                "query": expr,
                "wrong_query": wrong_expr,
                "raw_result": str(vol),
                "wrong_raw_result": str(wrong_vol),
                "extracted_val": f"{vol:,.2f} cm³",
                "wrong_extracted_val": f"{wrong_vol:,.2f} cm³",
                "extracted_dict": {"volume_cm3": vol},
                "wrong_extracted_dict": {"volume_cm3": wrong_vol}
            }]

    else:  # multi_hop
        template_choice = rng.choice(["company_hq_city_population", "company_valuation_stake", "scientist_birth_city_country", "element_block_mass"])

        if template_choice == "company_hq_city_population":
            comp = rng.choice(COMPANIES)
            matching_cities = [c for c in CITIES if c["city"] == comp["hq_city"]]
            city = matching_cities[0] if matching_cities else CITIES[0]

            prompt = f"Find the headquarters city of {comp['name']}, and then determine the total population of that city."
            gt = f"{comp['name']} is headquartered in {comp['hq_city']}, which has a population of {city['population']:,} residents."
            wrong_ans = f"{comp['name']} is headquartered in {comp['wrong_hq']}, which has a population of {city['wrong_pop']:,} residents."
            plan = (f"Lookup HQ city of {comp['name']}", "Query municipal database for population", "Combine facts into final answer")
            hops = [
                {
                    "tool_name": "search",
                    "query": f"Where is the headquarters of {comp['name']} located?",
                    "wrong_query": f"Where was {comp['name']} founded?",
                    "raw_result": f"{comp['name']} corporate headquarters is located in {comp['hq_city']}, {comp['country']}.",
                    "wrong_raw_result": f"{comp['name']} corporate offices are located in {comp['wrong_hq']}, {comp['country']}.",
                    "extracted_val": comp["hq_city"],
                    "wrong_extracted_val": comp["wrong_hq"],
                    "extracted_dict": {"hq_city": comp["hq_city"]},
                    "wrong_extracted_dict": {"hq_city": comp["wrong_hq"]}
                },
                {
                    "tool_name": "database_lookup",
                    "query": f"SELECT population FROM cities WHERE city = '{comp['hq_city']}'",
                    "wrong_query": f"SELECT population FROM cities WHERE city = '{comp['wrong_hq']}'",
                    "raw_result": json.dumps({"city": comp["hq_city"], "population": city["population"]}),
                    "wrong_raw_result": json.dumps({"city": comp["wrong_hq"], "population": city["wrong_pop"]}),
                    "extracted_val": f"{city['population']:,} residents",
                    "wrong_extracted_val": f"{city['wrong_pop']:,} residents",
                    "extracted_dict": {"population": city["population"]},
                    "wrong_extracted_dict": {"population": city["wrong_pop"]}
                }
            ]

        elif template_choice == "company_valuation_stake":
            comp = rng.choice(COMPANIES)
            pct = rng.choice([2.5, 5.0, 7.5, 10.0])
            val_b = comp["valuation_b"]
            stake_val = round(val_b * (pct / 100.0), 2)
            wrong_stake_val = round(comp["wrong_val"] * (pct / 100.0), 2)

            prompt = f"What is the market valuation of {comp['name']} in billions, and what would a {pct}% equity stake be worth in billions of dollars?"
            gt = f"{comp['name']} has a valuation of ${val_b}B, so a {pct}% equity stake is worth ${stake_val:.2f}B."
            wrong_ans = f"{comp['name']} has a valuation of ${comp['wrong_val']}B, so a {pct}% equity stake is worth ${wrong_stake_val:.2f}B."
            plan = (f"Lookup valuation of {comp['name']}", f"Calculate {pct}% equity value", "Synthesize findings")
            hops = [
                {
                    "tool_name": "search",
                    "query": f"Market capitalization of {comp['name']}",
                    "wrong_query": f"Historical revenue of {comp['name']}",
                    "raw_result": f"{comp['name']} valuation is estimated at ${val_b} Billion USD.",
                    "wrong_raw_result": f"{comp['name']} valuation is estimated at ${comp['wrong_val']} Billion USD.",
                    "extracted_val": f"${val_b} Billion",
                    "wrong_extracted_val": f"${comp['wrong_val']} Billion",
                    "extracted_dict": {"valuation_b": val_b},
                    "wrong_extracted_dict": {"valuation_b": comp["wrong_val"]}
                },
                {
                    "tool_name": "calculator",
                    "query": f"{val_b} * ({pct} / 100)",
                    "wrong_query": f"{comp['wrong_val']} * ({pct} / 100)",
                    "raw_result": str(stake_val),
                    "wrong_raw_result": str(wrong_stake_val),
                    "extracted_val": f"${stake_val:.2f} Billion",
                    "wrong_extracted_val": f"${wrong_stake_val:.2f} Billion",
                    "extracted_dict": {"stake_val_b": stake_val},
                    "wrong_extracted_dict": {"stake_val_b": wrong_stake_val}
                }
            ]

        elif template_choice == "scientist_birth_city_country":
            sci = rng.choice(SCIENTISTS)
            prompt = f"In which city was {sci['name']} born, and what is the field of study associated with them?"
            gt = f"{sci['name']} was born in {sci['birth_city']}, and is celebrated for discoveries in {sci['field']}."
            wrong_ans = f"{sci['name']} was born in {sci['birth_city']}, and is celebrated for discoveries in {sci['wrong_field']}."
            plan = (f"Search biographical archive for {sci['name']}", "Query database for scientific field", "Produce profile")
            hops = [
                {
                    "tool_name": "search",
                    "query": f"Birthplace of {sci['name']}",
                    "wrong_query": f"Education of {sci['name']}",
                    "raw_result": f"{sci['name']} was born in {sci['birth_city']} in {sci['birth_year']}.",
                    "wrong_raw_result": f"{sci['name']} lived in {sci['birth_city']}.",
                    "extracted_val": sci["birth_city"],
                    "wrong_extracted_val": sci["birth_city"],
                    "extracted_dict": {"birth_city": sci["birth_city"]},
                    "wrong_extracted_dict": {"birth_city": sci["birth_city"]}
                },
                {
                    "tool_name": "database_lookup",
                    "query": f"SELECT field, discovery FROM scientists WHERE name = '{sci['name']}'",
                    "wrong_query": f"SELECT field FROM scientists WHERE name = 'UNKNOWN'",
                    "raw_result": json.dumps({"name": sci["name"], "field": sci["field"], "discovery": sci["discovery"]}),
                    "wrong_raw_result": json.dumps({"name": sci["name"], "field": sci["wrong_field"], "discovery": sci["wrong_disc"]}),
                    "extracted_val": f"Field: {sci['field']}; Discovery: {sci['discovery']}",
                    "wrong_extracted_val": f"Field: {sci['wrong_field']}; Discovery: {sci['wrong_disc']}",
                    "extracted_dict": {"field": sci["field"], "discovery": sci["discovery"]},
                    "wrong_extracted_dict": {"field": sci["wrong_field"], "discovery": sci["wrong_disc"]}
                }
            ]

        else:  # element_block_mass
            elem = rng.choice(ELEMENTS)
            vol_cm3 = rng.choice([25, 50, 100, 250])
            density = elem["density_g_cm3"]
            wrong_density = elem["wrong_den"]
            mass = round(density * vol_cm3, 2)
            wrong_mass = round(wrong_density * vol_cm3, 2)

            prompt = f"Find the density of {elem['element']} in g/cm³, and calculate the total mass in grams of a {vol_cm3} cm³ solid block."
            gt = f"{elem['element']} has a density of {density} g/cm³, resulting in a total mass of {mass:,.2f} grams for a {vol_cm3} cm³ block."
            wrong_ans = f"{elem['element']} has a density of {wrong_density} g/cm³, resulting in a total mass of {wrong_mass:,.2f} grams for a {vol_cm3} cm³ block."
            plan = (f"Lookup density of {elem['element']}", f"Multiply density by volume ({vol_cm3} cm³)", "Summarize mass")
            hops = [
                {
                    "tool_name": "database_lookup",
                    "query": f"SELECT density_g_cm3 FROM elements WHERE element = '{elem['element']}'",
                    "wrong_query": f"SELECT density FROM elements WHERE id = 0",
                    "raw_result": json.dumps({"element": elem["element"], "density_g_cm3": density}),
                    "wrong_raw_result": json.dumps({"element": elem["element"], "density_g_cm3": wrong_density}),
                    "extracted_val": f"{density} g/cm³",
                    "wrong_extracted_val": f"{wrong_density} g/cm³",
                    "extracted_dict": {"density": density},
                    "wrong_extracted_dict": {"density": wrong_density}
                },
                {
                    "tool_name": "calculator",
                    "query": f"{density} * {vol_cm3}",
                    "wrong_query": f"{wrong_density} * {vol_cm3}",
                    "raw_result": str(mass),
                    "wrong_raw_result": str(wrong_mass),
                    "extracted_val": f"{mass:,.2f} grams",
                    "wrong_extracted_val": f"{wrong_mass:,.2f} grams",
                    "extracted_dict": {"mass_grams": mass},
                    "wrong_extracted_dict": {"mass_grams": wrong_mass}
                }
            ]

    return TaskSpec(
        task_type=task_type,
        prompt=prompt,
        ground_truth_answer=gt,
        plausible_wrong_answer=wrong_ans,
        plan_steps=plan,
        hops=hops
    )

print("TaskSpec generator defined")
