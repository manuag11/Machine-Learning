N = 5; % Number of nodes in the graph
num_samples = 10000; % Number of samples to be used in MLE and EM
hiding_proportion = 0.5; % proportion of data to be hidden
max_iterations = 50; % Maximum number of iterations for EM
epsilon = 0.0000001; % Smoothing constant while computation of KL divergence

% Construct DAG
[DAG,A,B,C,D,E]=constructDAG(N);

nodes = 1:N;

% All variables can take value 0 or 1
size_nodes = 2*ones(1,N); 

% Constructing Bayesian Net from DAG
bayes_net = mk_bnet(DAG, size_nodes, 'discrete', nodes);
bayes_net.CPD{A} = tabular_CPD(bayes_net, A, [0.001 0.999]);
bayes_net.CPD{B} = tabular_CPD(bayes_net, B, [0.002 0.998]);
bayes_net.CPD{C} = tabular_CPD(bayes_net, C, [0.95 0.94 0.29 0.001 0.05 0.06 0.71 0.999]);
bayes_net.CPD{D} = tabular_CPD(bayes_net, D, [0.90 0.05 0.10 0.95]);
bayes_net.CPD{E} = tabular_CPD(bayes_net, E, [0.70 0.01 0.30 0.99]);

CPT = compute_conditional_probability(bayes_net, N);
dispcpt(CPT{5})

inference = cell(1,N);
Engine_1 = jtree_inf_engine(bayes_net);
[Engine_2, ~] = enter_evidence(Engine_1, inference);
marg_nodes = marginal_nodes(Engine_2, [B C]);
probability = reshape(marg_nodes.T, 1, 2^2) + epsilon

samples=generate_samples(N,num_samples,bayes_net);

% Initializing Bayes network
bayes_net1 = mk_bnet(DAG, size_nodes);
rand('state',0);
bayes_net1.CPD{A} = tabular_CPD(bayes_net1, A);
bayes_net1.CPD{B} = tabular_CPD(bayes_net1, B);
bayes_net1.CPD{C} = tabular_CPD(bayes_net1, C);
bayes_net1.CPD{D} = tabular_CPD(bayes_net1, D);
bayes_net1.CPD{E} = tabular_CPD(bayes_net1, E);
Engine_3 = jtree_inf_engine(bayes_net1);

% MLE_estimation
probability_MLE = MLE_estimation(bayes_net1,samples,N,A,B,C,D,E,epsilon)

% hiding values
new_samples = samples;
for j=1:num_samples
    if rand() < hiding_proportion
        new_samples{N, j} = [];
    end
end

% EM_estimation
[probability_EM LLtrace] = EM_estimation(Engine_3,new_samples,max_iterations,N,A,B,C,D,E,epsilon);

% Computing KL divergences for MLE and EM
possible_events = [1:2^2]';
KLdiv_MLE = compute_kl_div(possible_events, probability_MLE', probability')
KLdiv_EM = compute_kl_div(possible_events, probability_EM', probability') 

plot(LLtrace/num_samples,'o-');