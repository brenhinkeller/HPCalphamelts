if ~exist('mcigncn1','var')
    load mcigncn1
end
if ~exist('igncn1','var')
    load igncn1
end

calcelements={'SiO2','TiO2','Al2O3','Fe2O3T','Cr','FeOT','MnO','MgO','Ni','Co','CaO','Na2O','K2O','P2O5','CO2','H2O_Total'};
nbins = 20;

agemin = 2500;
agemax = 4000; 

means=NaN(length(calcelements),nbins);
t = mcigncn1.Age>agemin&mcigncn1.Age<agemax;
for i=1:length(calcelements)
    [~, means(i,:), ~]=bin(mcigncn1.SiO2(t),mcigncn1.(calcelements{i})(t),40,80,length(mcigncn1.SiO2)./length(igncn1.SiO2),nbins);
end

iCr=find(cellfun(@(x)strcmp(x,'Cr'), calcelements));
iNi=find(cellfun(@(x)strcmp(x,'Ni'), calcelements));
iCo=find(cellfun(@(x)strcmp(x,'Co'), calcelements));

means(iCr,:)=means(iCr,:)*(51.9961+24)/51.9961/10^4;
means(iNi,:)=means(iNi,:)*(58.6934+16)/58.6934/10^4;
means(iCo,:)=means(iCo,:)*(58.9332+16)/58.9332/10^4;

startingindex = 5;
means=means';
ic=means(startingindex,:);
means(:,15:16)=[]; % Get rid of CO2 and H2O
means(:,4)=[]; % Get rid of Fe2O3
ic(4)=0; % Zero-out initial 

composition=means(startingindex+1:end,:); 



fprintf('	const double sc[%i]={',length(ic));
for j=1:length(ic)
    fprintf('%f',ic(j))
    if j<length(ic)
        fprintf(',')
    else
        fprintf('};\n\n')
    end
end

fprintf('	const double composition[%i][%i] = \n		{{',size(composition,1),size(composition,2))
for i=1:size(composition,1)
    for j=1:size(composition,2)
        fprintf('%f',composition(i,j))
        if j<size(composition,2)
            fprintf(',')
        else
            fprintf('}')
        end
    end
    if i<size(composition,1)
        fprintf('\n		{')
    else
        fprintf('};\n')
    end
end