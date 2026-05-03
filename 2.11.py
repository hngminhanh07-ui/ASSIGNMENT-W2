!pip install scikit-fuzzy
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
import matplotlib.pyplot as plt

distance = ctrl.Antecedent(np.arange(0, 51, 1), 'distance')
traffic = ctrl.Antecedent(np.arange(0, 101, 1), 'traffic')
demand = ctrl.Antecedent(np.arange(0, 101, 1), 'demand')
weather = ctrl.Antecedent(np.arange(0, 101, 1), 'weather')
rating = ctrl.Antecedent(np.arange(1, 5.1, 0.1), 'rating')
punctuality = ctrl.Antecedent(np.arange(0, 101, 1), 'punctuality')
price = ctrl.Consequent(np.arange(0, 101, 1), 'price')
reward = ctrl.Consequent(np.arange(0, 101, 1), 'reward')

distance['short'] = fuzz.trimf(distance.universe, [0, 0, 3])
distance['medium'] = fuzz.trimf(distance.universe, [2, 8, 20])
distance['long'] = fuzz.trimf(distance.universe, [6, 20, 50])
distance['very_long'] = fuzz.trimf(distance.universe, [15, 35, 50])

traffic['low'] = fuzz.trimf(traffic.universe, [0, 0, 30])
traffic['medium'] = fuzz.trimf(traffic.universe, [20, 50, 70])
traffic['high'] = fuzz.trimf(traffic.universe, [60, 100, 100])

demand['low'] = fuzz.trimf(demand.universe, [0, 0, 30])
demand['medium'] = fuzz.trimf(demand.universe, [20, 50, 70])
demand['high'] = fuzz.trimf(demand.universe, [60, 100, 100])

weather['good'] = fuzz.trimf(weather.universe, [0, 0, 30])
weather['moderate'] = fuzz.trimf(weather.universe, [20, 50, 80])
weather['bad'] = fuzz.trimf(weather.universe, [60, 100, 100])

rating['poor'] = fuzz.trimf(rating.universe, [1, 1, 2.5])
rating['average'] = fuzz.trimf(rating.universe, [2, 3.5, 4])
rating['good'] = fuzz.trimf(rating.universe, [3.5, 5, 5])

punctuality['late'] = fuzz.trimf(punctuality.universe, [0, 0, 50])
punctuality['on_time'] = fuzz.trimf(punctuality.universe, [40, 60, 80])
punctuality['early'] = fuzz.trimf(punctuality.universe, [70, 100, 100])

price['low'] = fuzz.trimf(price.universe, [0, 0, 35])
price['medium'] = fuzz.trimf(price.universe, [25, 50, 75])
price['high'] = fuzz.trimf(price.universe, [60, 80, 100])
price['very_high'] = fuzz.trimf(price.universe, [80, 100, 100])

reward['none'] = fuzz.trimf(reward.universe, [0, 0, 25])
reward['few'] = fuzz.trimf(reward.universe, [10, 35, 55])
reward['moderate'] = fuzz.trimf(reward.universe, [40, 60, 80])
reward['high'] = fuzz.trimf(reward.universe, [70, 100, 100])

distance.view()
traffic.view()
demand.view()
weather.view()
rating.view()
punctuality.view()
price.view()
reward.view()

rule1 = ctrl.Rule(distance['short'] & traffic['low'] & demand['low'], price['low'])
rule2 = ctrl.Rule(distance['short'] & traffic['medium'] & demand['high'], price['medium'])
rule3 = ctrl.Rule(distance['medium'] & traffic['high'] & demand['high'], price['high'])
rule4 = ctrl.Rule(distance['long'] & traffic['medium'] & weather['good'], price['medium'])
rule5 = ctrl.Rule(distance['long'] & traffic['high'] & weather['bad'], price['very_high'])
rule6 = ctrl.Rule(distance['very_long'] & traffic['high'] & demand['high'], price['very_high'])
rule7 = ctrl.Rule(distance['medium'] & traffic['low'] & demand['low'], price['medium'])
rule8 = ctrl.Rule(distance['short'] & demand['high'] & weather['bad'], price['high'])
rule9 = ctrl.Rule(distance['very_long'] & weather['bad'], price['very_high'])
rule10 = ctrl.Rule(distance['medium'] & traffic['medium'] & weather['moderate'], price['medium'])
rule11 = ctrl.Rule(rating['good'] & punctuality['early'], reward['high'])
rule12 = ctrl.Rule(rating['average'] & punctuality['on_time'], reward['moderate'])
rule13 = ctrl.Rule(rating['poor'] & punctuality['late'], reward['none'])
rule14 = ctrl.Rule(distance['long'] & traffic['high'] & punctuality['on_time'], reward['high'])
rule15 = ctrl.Rule(distance['medium'] & traffic['medium'] & rating['good'], reward['moderate'])
rule16 = ctrl.Rule(rating['poor'] & punctuality['late'], reward['none'])
rule17 = ctrl.Rule(distance['very_long'] & weather['bad'] & rating['good'], reward['high'])
rule18 = ctrl.Rule(distance['short'] & rating['average'] & punctuality['on_time'], reward['few'])
rule19 = ctrl.Rule(distance['long'] & traffic['high'] & punctuality['late'], reward['few'])
rule20 = ctrl.Rule(distance['medium'] & weather['moderate'] & rating['good'], reward['moderate'])

rule=[rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19, rule20]

grab_ctrl = ctrl.ControlSystem(rule)
grab_sim = ctrl.ControlSystemSimulation(grab_ctrl)

grab_sim.input['distance'] = 27
grab_sim.input['traffic'] = 70
grab_sim.input['demand'] = 45
grab_sim.input['weather'] = 80
grab_sim.input['rating'] = 4.5
grab_sim.input['punctuality'] = 100

price.view(sim=grab_sim)
reward.view(sim=grab_sim)
plt.show()
