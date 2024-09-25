opt_prompt = '''
Your task is to explain, compare various procurement scenarios and evaluate the optimal cost for a {triggerkeyword} for a energy demand of {demand} kwh timeframe = {timeframe}.

Utilize the provided purchasing scenarios, to compute and compare costs under specific constraints. Ensure adherence to the procurement limitations and residual demand beyond the specified limits should be satisfied
from the Spot Market.

Constraints and Pricing:
- Procure a maximum of 2400 kWh for renewable sources at 60 cents per kWh.
- Procure a maximum of 1800 kWh for Demand Side Response (DSR) at 35 cents per kWh.
- Procure a maximum of 1400 kWh for Battery at 70 cents per kwh
- Procure Rest of the demand fulfilled from Spot Market at 65 cents per kwh

Purchasing Policy - Allowed Scenarios:

{optimization_scenario}

Use below Cost Computing example formula for a Scenario for your training
for a particular procurement = min((total demand * Allowed Percentage), max demand limit from Limitation) = min value * each cost for that Procurement = cost for that Procurement
leftover demand = minimum utilized demand under each procurement
leftover demand cost = ( Total demand - leftover demand ) * Spot Market Price from Limitation
Total Cost = Total cost Procured for each + leftover demand cost

Present the results in 3 sections in your response
1. Compute the cost for each scenario by referring the example. 
2. Compare the different scenarios.
3. Conclude on the optimized cost and optimal scenario .

Ensure to compute the cost of each procurement correctly and
emphasize adherence to the specified limitations, especially the utilization of spot Market to fulfill the remaining demand after meeting the limits for other procurement.
'''

format_prompt = '''
Your Primary task is to format given input text enclosed in backticks ( ` ` ) containing calculated scenarios of energy consumption,
into a valid structured JSON format. JSON should contain List of Scenarios with each object having 3 Keys - 'scenario' - Scenario Name, 'calculation' - Detailed calculation for the Scenario (not in short form),
'cost' - Total cost of that scenario'.\n Success Example - {s_example}, Failure Example - {f_example}.
Input Text : `{input_text}`
'''
