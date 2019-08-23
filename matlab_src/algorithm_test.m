clear
clc
zero_mat = zeros(100, 1)*10;
Mpoints = rand(100, 2);
Mpoints = [Mpoints, zero_mat];
Fpoints = Mpoints + 10;
 registration(Mpoints, Fpoints);