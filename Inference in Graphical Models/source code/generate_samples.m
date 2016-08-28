function generated_samples = generate_samples(N,num_samples,bayes_net)
% This function generates generated_samples from a Bayes Net and returns them
generated_samples = cell(N, num_samples);
for i=1:num_samples
  generated_samples(:,i) = sample_bnet(bayes_net);
end

end

