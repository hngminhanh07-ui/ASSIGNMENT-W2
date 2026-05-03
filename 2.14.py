!pip install scikit-fuzzy

import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
import matplotlib.pyplot as plt

order_density = ctrl.Antecedent(np.arange(0, 101, 1), 'order_density')
delivery_urgency = ctrl.Antecedent(np.arange(0, 101, 1), 'delivery_urgency')
driver_load = ctrl.Antecedent(np.arange(0, 101, 1), 'driver_load')
traffic_conditions = ctrl.Antecedent(np.arange(0, 101, 1), 'traffic_conditions')
profit_per_delivery = ctrl.Antecedent(np.arange(0, 101, 1), 'profit_per_delivery')
orders_to_combine = ctrl.Consequent(np.arange(0, 11, 1), 'orders_to_combine')  
delivery_priority = ctrl.Consequent(np.arange(0, 101, 1), 'delivery_priority')  

order_density['low'] = fuzz.trimf(order_density.universe, [0, 0, 40])
order_density['medium'] = fuzz.trimf(order_density.universe, [30, 50, 70])
order_density['high'] = fuzz.trimf(order_density.universe, [60, 100, 100])

delivery_urgency['low'] = fuzz.trimf(delivery_urgency.universe, [0, 0, 40])
delivery_urgency['medium'] = fuzz.trimf(delivery_urgency.universe, [30, 50, 70])
delivery_urgency['high'] = fuzz.trimf(delivery_urgency.universe, [60, 100, 100])

driver_load['low'] = fuzz.trimf(driver_load.universe, [0, 0, 40])
driver_load['medium'] = fuzz.trimf(driver_load.universe, [30, 50, 70])
driver_load['high'] = fuzz.trimf(driver_load.universe, [60, 100, 100])

traffic_conditions['low'] = fuzz.trimf(traffic_conditions.universe, [0, 0, 40])
traffic_conditions['medium'] = fuzz.trimf(traffic_conditions.universe, [30, 50, 70])
traffic_conditions['high'] = fuzz.trimf(traffic_conditions.universe, [60, 100, 100])

profit_per_delivery['low'] = fuzz.trimf(profit_per_delivery.universe, [0, 0, 40])
profit_per_delivery['medium'] = fuzz.trimf(profit_per_delivery.universe, [30, 50, 70])
profit_per_delivery['high'] = fuzz.trimf(profit_per_delivery.universe, [60, 100, 100])

orders_to_combine['few'] = fuzz.trimf(orders_to_combine.universe, [0, 0, 3])
orders_to_combine['some'] = fuzz.trimf(orders_to_combine.universe, [2, 4, 6])
orders_to_combine['many'] = fuzz.trimf(orders_to_combine.universe, [5, 8, 10])

delivery_priority['low'] = fuzz.trimf(delivery_priority.universe, [0, 0, 40])
delivery_priority['medium'] = fuzz.trimf(delivery_priority.universe, [30, 50, 70])
delivery_priority['high'] = fuzz.trimf(delivery_priority.universe, [60, 100, 100])

rule_combine1 = ctrl.Rule(order_density['high'] & driver_load['low'] & traffic_conditions['low'], orders_to_combine['many'])
rule_combine2 = ctrl.Rule(order_density['medium'] & traffic_conditions['high'] & delivery_urgency['medium'], orders_to_combine['some'])
rule_combine3 = ctrl.Rule(driver_load['high'] & order_density['high'] & profit_per_delivery['medium'], orders_to_combine['some'])
rule_combine4 = ctrl.Rule(order_density['low'] & delivery_urgency['high'] & traffic_conditions['medium'], orders_to_combine['some'])
rule_combine5 = ctrl.Rule(profit_per_delivery['high'] & delivery_urgency['high'] & traffic_conditions['high'], orders_to_combine['some'])

rule_priority1 = ctrl.Rule(delivery_urgency['high'] & profit_per_delivery['high'], delivery_priority['high'])
rule_priority2 = ctrl.Rule(delivery_urgency['medium'] & traffic_conditions['medium'], delivery_priority['medium'])
rule_priority3 = ctrl.Rule(delivery_urgency['low'] & order_density['high'] & profit_per_delivery['low'], delivery_priority['low'])

combine_ctrl = ctrl.ControlSystem([rule_combine1, rule_combine2, rule_combine3, rule_combine4, rule_combine5, rule_priority1, rule_priority2, rule_priority3])
combine_sim = ctrl.ControlSystemSimulation(combine_ctrl)

combine_sim.input['order_density'] = 85
combine_sim.input['delivery_urgency'] = 50
combine_sim.input['driver_load'] = 20
combine_sim.input['traffic_conditions'] = 50
combine_sim.input['profit_per_delivery'] = 50
combine_sim.compute()

orders_to_combine.view(sim=combine_sim)
delivery_priority.view(sim=combine_sim)
plt.show()