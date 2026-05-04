!pip install scikit-fuzzy

import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
import matplotlib.pyplot as plt
product_demand = ctrl.Antecedent(np.arange(0, 101, 1), 'product_demand')
competitor_pricing_pressure = ctrl.Antecedent(np.arange(0, 101, 1), 'competitor_pricing_pressure')
store_reputation = ctrl.Antecedent(np.arange(1, 6, 0.1), 'store_reputation')
profit_margin = ctrl.Antecedent(np.arange(0, 101, 1), 'profit_margin')
seasonal_demand = ctrl.Antecedent(np.arange(0, 101, 1), 'seasonal_demand')
discount_percentage = ctrl.Consequent(np.arange(0, 71, 1), 'discount_percentage')

product_demand['low'] = fuzz.trimf(product_demand.universe, [0, 0, 40])
product_demand['medium'] = fuzz.trimf(product_demand.universe, [30, 50, 70])
product_demand['high'] = fuzz.trimf(product_demand.universe, [60, 100, 100])

competitor_pricing_pressure['low'] = fuzz.trimf(competitor_pricing_pressure.universe, [0, 0, 35])
competitor_pricing_pressure['medium'] = fuzz.trimf(competitor_pricing_pressure.universe, [25, 50, 75])
competitor_pricing_pressure['high'] = fuzz.trimf(competitor_pricing_pressure.universe, [65, 100, 100])


store_reputation['low'] = fuzz.trimf(store_reputation.universe, [1, 2, 4])
store_reputation['medium'] = fuzz.trimf(store_reputation.universe, [4, 4.25, 4.5])
store_reputation['high'] = fuzz.trimf(store_reputation.universe, [4.5, 4.75, 5])

profit_margin['low'] = fuzz.trimf(profit_margin.universe, [0, 0, 35])
profit_margin['medium'] = fuzz.trimf(profit_margin.universe, [25, 50, 75])
profit_margin['high'] = fuzz.trimf(profit_margin.universe, [65, 100, 100])

seasonal_demand['none'] = fuzz.trimf(seasonal_demand.universe, [0, 0, 30])
seasonal_demand['moderate'] = fuzz.trimf(seasonal_demand.universe, [20, 50, 80])
seasonal_demand['high'] = fuzz.trimf(seasonal_demand.universe, [60, 100, 100])


discount_percentage['very_low'] = fuzz.trimf(discount_percentage.universe, [0, 2.5, 5])
discount_percentage['low'] = fuzz.trimf(discount_percentage.universe, [5, 7.5, 10])
discount_percentage['medium'] = fuzz.trimf(discount_percentage.universe, [10, 15, 20])
discount_percentage['high'] = fuzz.trimf(discount_percentage.universe, [20, 30, 40])
discount_percentage['very_high'] = fuzz.trimf(discount_percentage.universe, [40, 55, 70])

rule1 = ctrl.Rule(product_demand['high'] & competitor_pricing_pressure['low'] & profit_margin['low'], discount_percentage['very_low'])
rule2 = ctrl.Rule(product_demand['low'] & competitor_pricing_pressure['high'] & profit_margin['high'], discount_percentage['high'])
rule3 = ctrl.Rule(store_reputation['high'] & profit_margin['medium'] & seasonal_demand['high'], discount_percentage['medium'])
rule4 = ctrl.Rule(competitor_pricing_pressure['high'] & seasonal_demand['high'] & profit_margin['high'], discount_percentage['very_high'])
rule5 = ctrl.Rule(store_reputation['low'] & product_demand['medium'] & profit_margin['low'], discount_percentage['medium'])
rule6 = ctrl.Rule(product_demand['high'] & seasonal_demand['none'] & competitor_pricing_pressure['low'], discount_percentage['very_low'])
rule7 = ctrl.Rule(profit_margin['high'] & competitor_pricing_pressure['medium'] & seasonal_demand['moderate'], discount_percentage['medium'])

rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7]
discount_ctrl = ctrl.ControlSystem(rules)
discount_sim = ctrl.ControlSystemSimulation(discount_ctrl)

discount_sim.input['product_demand'] = 85               
discount_sim.input['competitor_pricing_pressure'] = 50 
discount_sim.input['store_reputation'] = 4.2           
discount_sim.input['profit_margin'] = 80               
discount_sim.input['seasonal_demand'] = 50             

discount_sim.compute()
print(f"Mức chiết khấu tính toán được là: {discount_sim.output['discount_percentage']:.2f}%")
discount_percentage.view(sim=discount_sim)
plt.title('Ket qua tinh toan Chiet khau (Discount)')
plt.show()
rule3 = ctrl.Rule(store_reputation['high'] & profit_margin['medium'] & seasonal_demand['high'], discount_percentage['medium'])
rule4 = ctrl.Rule(competitor_pricing_pressure['high'] & seasonal_demand['high'] & profit_margin['high'], discount_percentage['very_high'])
rule5 = ctrl.Rule(store_reputation['low'] & product_demand['medium'] & profit_margin['low'], discount_percentage['medium'])
