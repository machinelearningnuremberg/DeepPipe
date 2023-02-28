from scipy.stats import norm as norm
import numpy as np


def EI(mean, sigma, best_f, epsilon = 0):    
    with np.errstate(divide='warn'):
        imp = mean -best_f - epsilon
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei

def random_suggest(model, X_obs, X_pen, y_obs, eval_in_batches=False):

    next_q = np.random.randint(1,len(X_pen))

    return next_q


def observe_and_suggest(model, X_obs, X_pen, y_obs, eval_in_batches=False):

    best_f = max(y_obs)
    model.fit(X_obs, y_obs)
    
    N = len(X_pen)
    
    if eval_in_batches:
        ei = []
        for i in range(100, N+100,100):
            mean, std = model.predict(X_pen[i-100:min(i,N)])
            temp_ei = EI(mean.reshape(-1), std.reshape(-1), best_f = best_f)
            ei+=temp_ei.tolist()

    else:
        mu, std = model.predict(X_pen)
        ei = EI(mu, std, best_f).tolist()    

    next_q = np.argmax(ei)

    return next_q

def create_BO (observe_and_suggest):

    def BO(model, Lambdas, response, ix_observed, ix_pending, bo_iter, eval_in_batches = True, return_observed = False):


        y = response[ix_observed]
        best = max(response)

        nan_ids = np.where(np.isnan(y))[0]
        y[nan_ids] = np.max(y)

        regret = [(best-max(y[0:(i+1)])).item() for i in range(0,len(y)) ]
        best_f = max(y)

        #to avoid different lenghts in tasks with nan values in init-ids
        regret += [min(regret) for i in range(5-len(regret))]

        for i in range(bo_iter):

            if best_f == best:
                regret+=[0]*(bo_iter-i)
                break

            next_q = observe_and_suggest(model, Lambdas[ix_observed], Lambdas[ix_pending], response[ix_observed], eval_in_batches)
            next_q = ix_pending[next_q]
            ix_observed.append(next_q)
            ix_pending.remove(next_q)

            best_f = max(response[ix_observed])
            regret.append((best-best_f).item())

        if return_observed: 
            
            return regret, ix_observed

        else:
            return regret

    return BO
