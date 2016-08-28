function [probability_EM, LLtrace] = EM_estimation(Engine_3,new_samples,max_iterations,N,A,B,C,D,E,epsilon)
% This function estimates the EM probability

[bayes_net3, LLtrace, Engine_6] = learn_params_em(Engine_3, new_samples, max_iterations);
CPT4 = compute_conditional_probability(bayes_net3, N);
dispcpt(CPT4{5})

inference = cell(1,N);
[Engine_7, ll] = enter_evidence(Engine_6, inference);
marg_nodes = marginal_nodes(Engine_7, [B C]);
probability_EM = reshape(marg_nodes.T, 1, 2^2) + epsilon

end