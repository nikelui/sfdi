% Description: SFD_depth_calculator determines spatial frequency domain (SFD)
%  penetration depth estimates using a scaled lookup table.  The table is
%  populated with results from Monte Carlo simulations at incremental
%  mus'/mua and spatial frequency (fx) values. 
%  
% Inputs:
%  mus' = reduced scattering coefficient [/mm] of the tissue of interest
%  mua  = absorption coefficient [/mm] of the tissue of interest
%  fxs  = spatial frequencies of interest
%
% Outputs:
%  depths = 2D matrix of penetration depth estimates, size [5, length(fxs)]
%           where first index is X (definition below) with values [10 25 50 75 90]
%
% The penetration depth estimates are determined by calculating the maximum
% visitation depths of detected photons.  This forms a function P_zmax that
% when integrated with respect to z to a depth d produces 
%   P_zmax(z<=d) = int_0^d P_zmax(z)dz.
% If d=positive infinity, then the integration results in total diffuse 
% reflectance, Rd.  Division of P_zmax by Rd
%   X = P_zmax(z<=d) / Rd
% provides the fraction (X) of the detected light that visited tissue depths d
% or less. Details are in accompanying manuscript.
%
% Author: Carole Hayakawa
% Date: 1/17/18
%
function [depths] = SFD_depth_calculator(mua, musp, fxs)
% cdf level tables
cdflevels=[10 25 50 75 90];
% set up return variable, depths to be of dimension [cdflevel, fxs]
depths=zeros(size(cdflevels,2),length(fxs));
% musp/mua and fxs used in table generation
tablemuspmua=[1 1.6 2 3 4 5 8 10 16 20 30 50 80 100 160 250 300 1000]; % MC mus'/mua 
tablefxs=[0 0.01 0.02 0.025 0.03 0.04 0.05 0.06 0.07 0.075 0.08 0.09 0.1 0.12 0.125 0.14 0.15 0.16 0.175 0.18 0.2 0.25 0.3 0.5 0.7]; % MC fx 
% determine input mus'/mua and lstar
muspmua=musp/mua;
lstar=1/(mua+musp);
% sanity check user input and display warnings if needed
if ((mua<0) || (musp<0))
  disp('WARNING: The input mua and musp values need to be >0 -> NaN results');
end
if (any(fxs<0))
  disp('WARNING: The input fxs need to be >0 -> NaN results');
end
if (max(fxs)>max(tablefxs/lstar))
  disp('WARNING: The input fxs need to be <0.7/l* -> NaN results');
end
if (muspmua<min(tablemuspmua))
  disp('WARNING: The input musp/mua nees to be >1 -> NaN results');
end
if (muspmua>max(tablemuspmua))
  disp('WARNING: The input musp/mua nees to be <1000 -> NaN results');
end
% load each cdf level table
for i=1:size(cdflevels,2)
  % code to read *.csv files
  table=csvread(sprintf('../data/cdflevel%dtable.csv',cdflevels(i)));
  % scale table entries and table fxs by lstar
  % then interpolate into table with input musp/mua and fxs
  depths(i,:)=interp2(tablemuspmua,tablefxs/lstar,table'*lstar,muspmua,fxs);
end
