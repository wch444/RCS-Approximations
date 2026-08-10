function hit = identify_hit(A,B,k)
%IDENTIFY_HIT Return cumulative Hit@1:K for a chosen item and ranked list.
%   A is the chosen item index; B lists predicted item indices in rank order.
    hit = zeros(k,1);
    a_1 = length(B);

    for i = 1:k
        if i<= a_1
            hit(i) = ismember(A,B(1:i));
        else
            hit(i) = hit(i-1);
        end
    end
