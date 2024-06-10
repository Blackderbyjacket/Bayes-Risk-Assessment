import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from networkx.drawing.nx_pydot import graphviz_layout

def define_cpd(variable, variable_card, values, state_names, evidence=None, evidence_card=None):
    # Define a CPD for a given variable with its states and (optional) evidence
    return TabularCPD(variable=variable, variable_card=variable_card, values=values, 
                      state_names=state_names, evidence=evidence, evidence_card=evidence_card)

def main():
    # Define the structure of the Bayesian Network
    model = BayesianNetwork([('Transactions', 'Transaction Value'), 
                             ('Transactions', 'Occurrence of Fraud'), 
                             ('Staff Experience', 'Occurrence of Fraud'),
                             ('Transaction Value', 'Impact'),
                             ('Occurrence of Fraud', 'Impact')])

    # Define the CPDs for each node in the network
    cpd_transactions = define_cpd('Transactions', 3, [[0.7], [0.1], [0.2]],
                                  {'Transactions': ['Low', 'Medium', 'High']})

    cpd_staff_exp = define_cpd('Staff Experience', 3, [[0.1], [0.6], [0.3]],
                               {'Staff Experience': ['Low', 'Medium', 'High']})

    cpd_trans_value = define_cpd('Transaction Value', 3, 
                                 [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]],
                                 {'Transaction Value': ['Low', 'Medium', 'High'],
                                  'Transactions': ['Low', 'Medium', 'High']},
                                 evidence=['Transactions'], evidence_card=[3])

    cpd_occurrence = define_cpd('Occurrence of Fraud', 2, 
                                [[0.99, 0.98, 0.97, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70],
                                 [0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]],
                                {'Occurrence of Fraud': ['False', 'True'],
                                 'Transactions': ['Low', 'Medium', 'High'],
                                 'Staff Experience': ['Low', 'Medium', 'High']},
                                evidence=['Transactions', 'Staff Experience'], evidence_card=[3, 3])

    cpd_impact = define_cpd('Impact', 4, 
                            [[0.96, 0.96, 0.96, 0.96, 0.96, 0.96],
                             [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                             [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                             [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]],
                            {'Impact': ['0-0', '0-100', '100-1000', '1000-10000'],
                             'Transaction Value': ['Low', 'Medium', 'High'],
                             'Occurrence of Fraud': ['False', 'True']},
                            evidence=['Transaction Value', 'Occurrence of Fraud'], evidence_card=[3, 2])

    # Add the CPDs to the Bayesian Network
    model.add_cpds(cpd_transactions, cpd_staff_exp, cpd_trans_value, cpd_occurrence, cpd_impact)

    # Check if the model is valid
    if model.check_model():
        print("Model is valid.")
    else:
        print("Model is invalid.")

    # Perform inference
    infer = VariableElimination(model)
    impact_prob = infer.query(variables=['Impact'], evidence={'Transaction Value': 'Medium', 'Transactions': 'Low'})
    print(impact_prob)

    # Visualization of the Bayesian Network using pydot and graphviz
    pos = graphviz_layout(model, prog="dot")
    plt.figure(figsize=(12, 8))
    nx.draw(model, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=15, font_weight='bold', arrows=True)
    plt.title('Bayesian Network Structure')
    plt.show()

if __name__ == "__main__":
    main()
