!pip install scikit-fuzzy

import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
import matplotlib.pyplot as plt

store_rating = ctrl.Antecedent(np.arange(1, 5.1, 0.1), 'store_rating')
sales_volume = ctrl.Antecedent(np.arange(0, 101, 1), 'sales_volume')
profit_margin = ctrl.Antecedent(np.arange(0, 101, 1), 'profit_margin')
seasonal_event = ctrl.Antecedent(np.arange(0, 101, 1), 'seasonal_event')
competitor_discount = ctrl.Antecedent(np.arange(0, 101, 1), 'competitor_discount')
discount_rate = ctrl.Consequent(np.arange(0, 71, 1), 'discount_rate')

store_rating['low'] = fuzz.trimf(store_rating.universe, [1, 1, 4.0])
store_rating['medium'] = fuzz.trimf(store_rating.universe, [4, 4.25, 4.5])
store_rating['high'] = fuzz.trimf(store_rating.universe, [4.5, 5, 5])

sales_volume['low'] = fuzz.trimf(sales_volume.universe, [0, 0, 35])
sales_volume['medium'] = fuzz.trimf(sales_volume.universe, [25, 50, 75])
sales_volume['high'] = fuzz.trimf(sales_volume.universe, [65, 100, 100])

profit_margin['low'] = fuzz.trimf(profit_margin.universe, [0, 0, 35])
profit_margin['medium'] = fuzz.trimf(profit_margin.universe, [25, 50, 75])
profit_margin['high'] = fuzz.trimf(profit_margin.universe, [65, 100, 100])

seasonal_event['none'] = fuzz.trimf(seasonal_event.universe, [0, 0, 30])
seasonal_event['moderate'] = fuzz.trimf(seasonal_event.universe, [20, 50, 80])
seasonal_event['high'] = fuzz.trimf(seasonal_event.universe, [60, 100, 100])

competitor_discount['low'] = fuzz.trimf(competitor_discount.universe, [0, 0, 35])
competitor_discount['medium'] = fuzz.trimf(competitor_discount.universe, [25, 50, 75])
competitor_discount['high'] = fuzz.trimf(competitor_discount.universe, [65, 100, 100])

discount_rate['very_low'] = fuzz.trimf(discount_rate.universe, [0, 0, 5])
discount_rate['low'] = fuzz.trimf(discount_rate.universe, [5, 7.5, 10])
discount_rate['medium'] = fuzz.trimf(discount_rate.universe, [10, 15, 20])
discount_rate['high'] = fuzz.trimf(discount_rate.universe, [20, 30, 40])
discount_rate['very_high'] = fuzz.trimf(discount_rate.universe, [40, 55, 70])

store_rating.view()
sales_volume.view()
profit_margin.view()
seasonal_event.view()
competitor_discount.view()
discount_rate.view()
plt.show()  

rule1 = ctrl.Rule(store_rating['high'] & sales_volume['high'] & profit_margin['high'], discount_rate['very_low'])
rule2 = ctrl.Rule(store_rating['low'] & sales_volume['low'] & profit_margin['high'], discount_rate['high'])
rule3 = ctrl.Rule(seasonal_event['high'] & competitor_discount['high'], discount_rate['very_high'])
rule4 = ctrl.Rule(store_rating['medium'] & sales_volume['medium'] & profit_margin['medium'], discount_rate['medium'])
rule5 = ctrl.Rule(competitor_discount['low'] & profit_margin['low'] & sales_volume['high'], discount_rate['very_low'])
rule6 = ctrl.Rule(store_rating['low'] & seasonal_event['none'], discount_rate['medium'])
rule7 = ctrl.Rule(sales_volume['low'] & profit_margin['low'], discount_rate['very_high'])

rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7]

discount_ctrl = ctrl.ControlSystem(rules)
discount_sim = ctrl.ControlSystemSimulation(discount_ctrl)

discount_sim.input['store_rating'] = 4.3
discount_sim.input['sales_volume'] = 45
discount_sim.input['profit_margin'] = 20
discount_sim.input['seasonal_event'] = 85
discount_sim.input['competitor_discount'] = 78
discount_sim.compute()

discount_rate.view(sim=discount_sim)
plt.show()
