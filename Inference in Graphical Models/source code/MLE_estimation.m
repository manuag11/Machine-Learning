function probability_MLE = MLE_estimation(bayes_net1,generated_samples,N,A,B,C,D,E,epsilon)
% This function estimates the MLE probability

bayes_net2 = learn_params(bayes_net1, generated_samples);
Engine_4 = jtree_inf_engine(bayes_net2);

CPT3 = compute_conditional_probability(bayes_net2, N);
dispcpt(CPT3{5})

inference = cell(1,N);
[Engine_5, ~] = enter_evidence(Engine_4, inference);
marg_nodes = marginal_nodes(Engine_5, [B C]);
probability_MLE = reshape(marg_nodes.T, 1, 2^2) + epsilon

end