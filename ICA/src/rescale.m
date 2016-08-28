function V_scaled = rescale(V,high,low)
%	SUMMARY rescales vector V to lie in the range low to high
maximum=max(V(:));
minimum=min(V(:));
V_scaled=(V-minimum)/(maximum-minimum)*(high-low)+low;
end

