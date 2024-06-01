from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

def define_cpd(variable, variable_card, values, state_names, evidence=None, evidence_card=None):
    return TabularCPD(variable=variable, variable_card=variable_card, values=values, 
                      state_names=state_names, evidence=evidence, evidence_card=evidence_card)

def main():
    model = BayesianNetwork([('Transactions', 'Transaction Value'), 
                             ('Transactions', 'Occurrence of Fraud'), 
                             ('Staff Experience', 'Occurrence of Fraud'),
                             ('Transaction Value', 'Impact'),
                             ('Occurrence of Fraud', 'Impact')])

    cpd_transactions = define_cpd('Transactions', 3, [[0.7], [0.1], [0.2]],
                                  {'Transactions': ['Low', 'Medium', 'High']})

    cpd_staff_exp = define_cpd('Staff Experience', 3, [[0.1], [0.6], [0.3]],
                               {'Staff Experience': ['Low', 'Medium', 'High']})

    cpd_trans_value = define_cpd('Transaction Value', 3, 
                                 [[0.5, 0.3, 0.2], [0.4, 0.4, 0.3], [0.1, 0.3, 0.5]],
                                 {'Transaction Value': ['Low', 'Medium', 'High'],
                                  'Transactions': ['Low', 'Medium', 'High']},
                                 evidence=['Transactions'], evidence_card=[3])

    cpd_occurrence = define_cpd('Occurrence of Fraud', 2, 
                                [[0.99, 0.97, 0.95, 0.97, 0.95, 0.93, 0.95, 0.93, 0.91],
                                 [0.01, 0.03, 0.05, 0.03, 0.05, 0.07, 0.05, 0.07, 0.09]],
                                {'Occurrence of Fraud': ['False', 'True'],
                                 'Transactions': ['Low', 'Medium', 'High'],
                                 'Staff Experience': ['Low', 'Medium', 'High']},
                                evidence=['Transactions', 'Staff Experience'], evidence_card=[3, 3])

    cpd_impact = define_cpd('Impact', 4, 
                            [[0.99, 0.98, 0.97, 0.96, 0.95, 0.94],
                             [0.005, 0.01, 0.015, 0.02, 0.025, 0.03],
                             [0.003, 0.005, 0.01, 0.015, 0.02, 0.025],
                             [0.002, 0.005, 0.005, 0.005, 0.005, 0.005]],
                            {'Impact': ['0-0', '0-100', '100-1000', '1000-10000'],
                             'Transaction Value': ['Low', 'Medium', 'High'],
                             'Occurrence of Fraud': ['False', 'True']},
                            evidence=['Transaction Value', 'Occurrence of Fraud'], evidence_card=[3, 2])

    model.add_cpds(cpd_transactions, cpd_staff_exp, cpd_trans_value, cpd_occurrence, cpd_impact)

    if model.check_model():
        print("Model is valid.")
    else:
        print("Model is invalid.")

    infer = VariableElimination(model)
    prob = infer.query(variables=['Impact'], evidence={'Transaction Value': 'High', 'Transactions': 'High'})
    print(prob)

if __name__ == "__main__":
    main()
