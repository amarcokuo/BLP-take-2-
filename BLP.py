import numpy as np
import scipy as sp
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import time
from numba import jit
import openturns as ot

class BLP:
    def __init__(self):

        # Load data
        df = pd.read_stata('sample.dta')

        # x1 variables enter the linear part of the estimation
        self.x1 = df.as_matrix(['var8', 'price', 'char1', 'char2'])

        # x2 variables enter the non-linear part
        self.x2 = df.as_matrix(['char1', 'char2'])

        # number of random coefficients
        self.nrc = 2

        # number of simulated "indviduals" per market
        self.ns = 200

        # number of markets, mktid = 1, 2, ..., 100

        self.nmkt = df['mktid'].max() - df['mktid'].min() + 1

        # number of brands per market. if the numebr differs by market this requires some "accounting" vector
        self.num_prod = df[['prodid', 'mktid']].groupby(['mktid']).agg(['count']).as_matrix()

        # this vector relates each observation to the market it is in
        self.cdid = np.kron(np.array([i for i in range(self.nmkt)], ndmin=2).T, np.ones((100, 1)))
        self.cdid = self.cdid.reshape(self.cdid.shape[0]).astype('int')

        ## this vector provides for each index the of the last observation
        ## in the data used here all brands appear in all markets. if this
        ## is not the case the two vectors, cdid and cdindex, have to be
        ## created in a different fashion but the rest of the program works fine.
        ## cdindex = [nbrn:nbrn:nbrn*nmkt]';
        self.cdindex = np.array([i for i in range((100 - 1), 100 * self.nmkt, 100)])

        # the market share of product j in market t
        self.s_jt = df.as_matrix(['sjt'])

        # the outside option share of product j in market t
        self.s_0t = df.as_matrix(['s0t'])

        # Load IV for the instruments and the x's.
        self.IV = df.as_matrix(['var8', 'char1', 'char2', 'z1', 'z2', 'z3', 'z4', 'z5', 'z6', 'z7', 'z8', 'z9', 'z10'])

        # create initial weight matrix
        self.invA = np.linalg.inv(self.IV.T @ self.IV)

        # The following codes compute logit results and save the mean utility as initial values for the search below

        # compute logit results and save the mean utility as initial values for the search below
        y = np.log(self.s_jt / self.s_0t)
        mid = self.x1.T @ self.IV @ self.invA @ self.IV.T
        self.init_theta = np.linalg.inv(mid @ self.x1) @ mid @ y
        self.old_delta = self.x1 @ self.init_theta
        self.old_delta_exp = np.exp(self.old_delta)

        self.V_names = ['Constant', 'Price', 'Char1', 'Char2']
        self.df_logit_iv = pd.DataFrame(self.init_theta, index=self.V_names)
        self.df_logit_iv.columns = ['Coef.']

        # create initial values for random coefficients, price, char1, char2
        # np.random.seed(1)
        # self.init_theta2 = np.absolute(np.random.normal(0, 1 , (self.nrc,1)))
        self.init_theta2 = np.ones((self.nrc, 1))

        # drawing from Halton sequence with self.ns draws across self.nrc, the number of random coefficients
        # x = halton_sequence(self.ns + 1, self.nrc)
        # v = np.asarray(x)

        # drawing from Halton sequence with self.ns draws across self.nrc, the number of random coefficients
        self.v = np.array(ot.HaltonSequence(self.nrc).generate(self.ns))
        # convert halton sequence into normal dist. draws
        self.vi = norm.ppf(self.v)


        self.gmmvalold = 0
        self.gmmdiff = 1

        self.iter = 0
        # self.theta2 = self.theta2_init
        self.delta = self.meanval(self.init_theta2)
        self.gmmresid = self.delta - self.x1 @ self.init_theta

    # Given the individuals' draws, this function computes the non-linear part of the utility, mu_ijt.

    def mktsh(self, old_delta_exp):
        # compute the market share for each product
        temp = self.ind_sh(old_delta_exp).T
        f = sum(temp) / float(self.ns)
        return f.T

    def ind_sh(self, old_delta_exp):
        try:
            theta2 = self.theta2
        except:
            theta2 = self.init_theta2

        self.expmu = np.exp(self.x2 @ ( theta2.reshape(self.nrc,1)* self.vi.T))
        # calculate exp(mu_ijt)

        eg = np.multiply(self.expmu, np.kron(np.ones((1, self.ns)), old_delta_exp))
        # for every product j at time t, exp(mu) * exp(delta) = exp(mu_ijt + delta_jt)


        gap = self.num_prod.flatten()
        # gap helps track the number of products in each market.
        # It would be very helpful if the number of products is changing over time
        z = np.zeros((self.nmkt, self.ns))
        start = 0
        end = 0
        for k in range(len(gap)):
            end += gap[k]
            # z[k, :] = sum(eg[start:end, :])
            z[k, :] = eg[start:end, :].sum(axis=0)
            start += gap[k]

        # denom = 1 / (1+sum(exp_j(delta_jt + mu_ijt)))
        denom = 1 / (1 + z)

        # expand the denom to all products
        denom = np.repeat(denom, gap, axis=0)

        # finally, return to individual (100 here) choice prob. for product jt at time t
        return np.multiply(eg, denom)



    @jit
    def meanval(self, theta2):

        n, k = self.x2.shape
        # self.expmu = np.exp(self.mufunc(theta2))
        self.theta2 = theta2
        if self.gmmdiff < 1e-6:
            etol = self.etol = 1e-15
        elif self.gmmdiff < 1e-3:
            etol = self.etol = 1e-13
        else:
            etol = self.etol = 1e-12
        norm = 1

        i = 0

        stepmin = 1
        stepmax = 1
        while norm > etol:
            # pred_s_jt = pred_s_jt.reshape(n, 1)
            # probably redundant
            g = self.s_jt / self.mktsh(self.old_delta_exp).reshape(n, 1)
            self.new_delta_exp = np.multiply(self.old_delta_exp, g)

            # Use SQUAREM method
            x = self.old_delta_exp
            x1 = self.new_delta_exp
            q1 = x1 - x
            x2 = np.multiply(x1, self.s_jt / self.mktsh(x1).reshape(n, 1))
            q2 = x2 - x1

            # Form quadratic terms
            sr2 = q1.T @ q1
            sq2 = np.sqrt(q2.T @ q2)
            sv2 = (q2 - q1).T @ (q2 - q1)
            srv = q1.T @ (q2 - q1)
            # Get the step-size
            alpha = np.sqrt(sr2 / sv2)
            # alpha = -srv/sv2
            # alpha = -sr2/srv
            alpha = max(stepmin, min(stepmax, alpha))
            xtemp = x + 2 * alpha * q1 + (alpha ** 2) * (q2 - q1)
            xnew = np.multiply(xtemp, self.s_jt / self.mktsh(xtemp).reshape(n, 1))
            if np.isnan(xnew).any():
                print("Error")
                xnew = x2

            if alpha == stepmax:
                stepmax = 4 * stepmax

            if alpha == stepmin and alpha < 0:
                stepmin = 4 * stepmin

            t = np.abs(xnew - xtemp)
            # t = np.abs(self.new_delta_exp - self.old_delta_exp)
            norm = np.mean(t)

            self.old_delta_exp = xnew
            # self.old_delta_exp = self.new_delta_exp
            i += 1
        # print ('# of iterations for delta convergence:', i)

        self.old_delta = np.log(self.old_delta_exp)
        return np.log(self.new_delta_exp)

    @jit
    def gmmobj(self, theta2):
        # compute GMM objective function
        self.delta = self.meanval(theta2)
        self.theta2 = theta2

        # the following deals with cases where the min algorithm drifts into region where the objective is not defined
        if max(np.isnan(self.delta)) == 1:
            f = 1e+10
        else:
            temp1 = self.x1.T @ self.IV
            temp2 = self.delta.T @ self.IV
            # self.theta1 = np.linalg.inv(temp1 @ self.invA @ temp1.T) @ temp1 @ self.invA @ temp2.T

            # using new weighting matrix
            self.theta1 = sp.linalg.solve(temp1 @ self.invA @ temp1.T, temp1 @ self.invA @ temp2.T)

            self.gmmresid = self.delta - self.x1 @ self.theta1

            temp3 = self.gmmresid.T @ self.IV

            f = temp3 @ self.invA @ temp3.T
            # using new weighting matrix

        self.gmmvalnew = f[0, 0]
        if self.gmmvalnew < self.gmmvalold:
            self.iter += 1

            # if self.iter % 10 ==0:
            # print ('# of valuations:', self.iter)
            # print ('gmm objective:', self.gmmvalnew)

        self.gmmdiff = np.abs(self.gmmvalold - self.gmmvalnew)
        self.gmmvalold = self.gmmvalnew
        return (f[0, 0])

    def jacob(self):
        # in the function, we want to generate a n*nrc, here it's 100,000*3 Jacobian matrix
        theta2 = self.theta2
        self.expmu = np.exp(self.x2 @ (theta2.reshape(self.nrc,1)* self.vi.T))

        shares = self.ind_sh(self.old_delta_exp)
        n, K = self.x2.shape
        v = self.vi
        num_prod = self.num_prod.flatten()
        f = np.zeros((n, K))
        share_theta2 = np.zeros((n, K))
        # computing JT*L matrix, which is (partial shares)/(partial theta2)
        for i in range(self.ns):
            v_s = np.zeros((n, K))
            x_xs = np.zeros((n, K))
            start = 0
            end = 0
            for t in range(blp.nmkt):
                end += num_prod[t]
                v_s[start:end, :] = np.multiply(v[i, :].reshape(1, K), shares[start:end, i].reshape(num_prod[t], 1))
                x_xs[start:end, :] = self.x2[start:end, :] - (
                self.x2[start:end, :].T @ shares[start:end, i].reshape(num_prod[t], 1)).T
                share_theta2[start:end, :] += np.multiply(v_s[start:end, :], x_xs[start:end, :])
                start += num_prod[t]
        share_theta2 = share_theta2 / self.ns

        # End of computing JT*L matrix, which is (partial shares)/(partial theta2)


        # computing JT*J matrix, which is (partial shares)/(partial delta), and D_delta Jacobian

        start = 0
        end = 0

        for t in range(self.nmkt):
            H = np.zeros((num_prod[t], num_prod[t]))
            end += num_prod[t]
            for i in range(self.ns):
                temp = shares[start:end, i].reshape(num_prod[t], 1)
                H1 = temp @ temp.T
                H += (np.diag(temp.flatten()) - H1) / self.ns

            H_inv = -np.linalg.inv(H)
            f[start:end, :] = H_inv @ share_theta2[start:end, :]
            start += num_prod[t]

        # End of computing D_delta Jacobian JT*L matrix

        return f

    def varcov(self):
        N = self.x1.shape[0]
        Z = self.IV.shape[1]

        Q = self.IV.T @ np.hstack((self.x1, self.jacob()))

        a = np.linalg.inv(Q.T @ self.invA @ Q)
        IVres = np.multiply(self.IV, self.gmmresid * np.ones((1, Z)))
        omega = IVres.T @ IVres

        f = a @ Q.T @ self.invA @ omega @ self.invA @ Q @ a
        return f

    def cal_se(self):
        varcov = self.varcov()
        se_all = np.sqrt(varcov.diagonal())
        return se_all

    def result(self, rex):

        rex = res.x
        D_names = ['Coef.', 'S.E.']
        V_names = ['constant', 'price', 'char1', 'char2',  'rc_char1', 'rc_char2']

        df3 = pd.DataFrame(np.hstack((self.theta1.flatten(), rex)), index=V_names)
        other_se = self.cal_se()

        df4 = pd.DataFrame(other_se, index=V_names)
        result = pd.concat([df3, df4], axis=1)
        result.columns = D_names
        self.result = result

        return result


if __name__ == '__main__':
    start_time = time.time()
    Nfeval = 1


    def callbackF(Xi):
        global Nfeval
        print('{:>10}  {:10.6f}'.format(Nfeval, blp.gmmobj(Xi)))
        Nfeval += 1



    blp = BLP()
    init_theta = blp.init_theta2
    print("---Linear parameters from logit-IV regresion:---")
    print(blp.df_logit_iv)
    print("---Start searching for random coefficients---" )

    print('{:>10}  {:>10}'.format('Iter', 'f(X)'))
    # gradient = blp.gradient_GMM()
    res = minimize(blp.gmmobj, init_theta, method='BFGS', callback=callbackF, options={'maxiter': 20, 'disp': True})
    print(blp.result(res.x))
    # Nelder-Mead
    print("--- This estimation used %s Halton draws. ---" % (blp.ns))

    print("--- %s seconds ---" % (time.time() - start_time))