from pyscipopt import Model


def scip_test():
    scip = Model()
    scip.readProblem('master.lp')
    scip.optimize()


if __name__ == '__main__':
    scip_test()
