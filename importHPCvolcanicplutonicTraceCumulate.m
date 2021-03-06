% Load all necessary data
if ~exist('mcvolcanic','var'); load mcvolcanic; end
if ~exist('volcanic','var'); load volcanic; end
if ~exist('mcplutonic','var'); load mcplutonic; end
if ~exist('plutonic','var'); load plutonic; end
if ~exist('p','var'); load partitioncoeffs; end
if ~exist('residuals','file') % If not processed, delete column headers from residuals file
    [~,~]=unix('grep n residuals.csv > residualcolumns');
    [~,~]=unix('sed ''/^n/d'' residuals.csv > residuals');
    [~,~]=unix('rm residuals.csv');
end
load residuals
residuals=sortrows(residuals,2); % Sort by column 2 (least squared residual)
residuals(residuals(:,2)==0,:)=[]; % Delete simulations that did not run (residual of zero)

% Trace elements to plot
traceelements={'Sr','Ba','Eu','Gd'};

xfigs=2;
yfigs=2;
nsims=200; 


% Plot original
figure;
for i=1:length(traceelements)
    hold on; subaxis(yfigs,xfigs,mod(i-1,xfigs*yfigs)+1,'SpacingVert',0.02,'SpacingHoriz',0.05,'Margin',0.06); binplot(mcplutonic.SiO2,mcplutonic.(traceelements{i}),40,80,length(mcplutonic.SiO2)./length(plutonic.SiO2),20,'.b')
    hold on; subaxis(yfigs,xfigs,mod(i-1,xfigs*yfigs)+1,'SpacingVert',0.02,'SpacingHoriz',0.05,'Margin',0.06); binplot(mcvolcanic.SiO2,mcvolcanic.(traceelements{i}),40,80,length(mcvolcanic.SiO2)./length(volcanic.SiO2),20,'.r')
    ylabel(traceelements{i}); setlabelfontsize(12)
    if any(mod(i-1,xfigs*yfigs)+1==(xfigs*yfigs-(0:xfigs-1))); xlabel('SiO2'); else set(gca,'xticklabel',[]); end
end
legend({'Plutonic','Volcanic'})

% Obtain trace element compotitions
startingcol=7;
tr=NaN(length(traceelements),20);
for i=1:length(traceelements)
    [~, tr(i,:), ~]=bin(mcvolcanic.SiO2,mcvolcanic.(traceelements{i}),40,80,length(mcvolcanic.SiO2)./length(volcanic.SiO2),20);
end

elements={'SiO2'};
ref.quartz.SiO2=100;
ref.apatite.CaO=55.82;
ref.apatite.P2O5=42.39;
ref.apatite.H2O=1.79;
ref.whitlockite.CaO=47.1;
ref.whitlockite.MgO=2.63;
ref.whitlockite.FeO=3.35;
ref.whitlockite.P2O5=46.37;
ref.aenigmatite.Na2O=6.73;
ref.aenigmatite.CaO=1.08;
ref.aenigmatite.MgO=1.08;
ref.aenigmatite.TiO2=9.77;
ref.aenigmatite.MnO=1.65;
ref.aenigmatite.Al2O3=1.90;
ref.aenigmatite.FeO=35.76;
ref.aenigmatite.Fe2O3=3.07;
ref.aenigmatite.SiO2=38.83;


% Colormap to use for plotting water contents
saturation=smooth([linspace(1,0.9,50)'; linspace(0.9,1,50)'],10);
hue=[linspace(1,0,100)'.^3,zeros(100,1),linspace(0,1,100)'];
cmap=hue.*repmat(saturation,1,3);



% Loop through all the melts simulations of interest
for ssim=1:nsims
    dir = ['out' num2str(residuals(ssim,1))];
    melts=struct;
    % Import the results
    if exist([dir '/Phase_main_tbl.txt'],'file')
        cells=importc([dir '/Phase_main_tbl.txt'],' ');
        emptycols=all(cellfun('isempty', cells),1);
        cells=cells(:,~emptycols);
        pos=[find(all(cellfun('isempty', cells),2)); size(cells,1)+1];
        melts.minerals=cell(length(pos)-1,1);
        for i=1:(length(pos)-1)
            name=varname(cells(pos(i)+1,1));
            melts.(name{1}).elements=varname(cells(pos(i)+2,~cellfun('isempty',cells(pos(i)+2,:))));
            melts.(name{1}).data=str2double(cells(pos(i)+3:pos(i+1)-1,1:length(melts.(name{1}).elements)));
            melts.(name{1})=elementify(melts.(name{1}));
            melts.minerals(i)=name;
        end
        
        
        % Determine minimum and maximum temperature in the simulation
        minT=NaN;
        maxT=NaN;
        for i=1:length(melts.minerals)
            maxT=max([maxT max(melts.(melts.minerals{i}).Temperature)]);
            minT=min([minT min(melts.(melts.minerals{i}).Temperature)]);
        end
        
        % Determine temperature step
        if range(melts.liquid_0.Temperature(1:end-1)-melts.liquid_0.Temperature(2:end))<10^-5
            deltaT=melts.liquid_0.Temperature(1)-melts.liquid_0.Temperature(2);
            simlength=round((maxT-minT)./deltaT)+1;
        else
            error('Non-uniform temperature step');
        end
        
        
        % Calculate individual components for feldspar and oxides
        An_Ca=(220.1298+56.18)/56.18;
        Ab_Na=(228.2335+30.99)/30.99;
        Or_K=(228.2335+47.1)/47.1;
        for i=1:length(melts.minerals)
            if ~isempty(strfind(lower(melts.minerals{i}),'feldspar')) && ~isfield(melts, 'anorthite')
                feldsparComp=bsxfun(@rdivide, [melts.(melts.minerals{i}).CaO*An_Ca melts.(melts.minerals{i}).Na2O*Ab_Na melts.(melts.minerals{i}).K2O*Or_K], (melts.(melts.minerals{i}).CaO*An_Ca + melts.(melts.minerals{i}).Na2O*Ab_Na + melts.(melts.minerals{i}).K2O*Or_K));
                melts.(['anorthite' melts.minerals{i}(9:end)]).mass=melts.(melts.minerals{i}).mass.*feldsparComp(:,1);
                melts.(['anorthite' melts.minerals{i}(9:end)]).Temperature=melts.(melts.minerals{i}).Temperature;
                melts.(['albite' melts.minerals{i}(9:end)]).mass=melts.(melts.minerals{i}).mass.*feldsparComp(:,2);
                melts.(['albite' melts.minerals{i}(9:end)]).Temperature=melts.(melts.minerals{i}).Temperature;
                melts.(['orthoclase' melts.minerals{i}(9:end)]).mass=melts.(melts.minerals{i}).mass.*feldsparComp(:,3);
                melts.(['orthoclase' melts.minerals{i}(9:end)]).Temperature=melts.(melts.minerals{i}).Temperature;
                melts.minerals=unique([melts.minerals; ['anorthite' melts.minerals{i}(9:end)]; ['albite' melts.minerals{i}(9:end)]; ['orthoclase' melts.minerals{i}(9:end)];],'stable');
            elseif ~isempty(strfind(lower(melts.minerals{i}),'rhmoxide'))
                Ilmenite=(melts.rhmoxide0.TiO2+melts.rhmoxide0.MnO+(melts.rhmoxide0.TiO2*(71.8444/79.8768)-melts.rhmoxide0.MnO*(71.8444/70.9374)))/100;
                Magnetite=(melts.rhmoxide0.FeO-(melts.rhmoxide0.TiO2-melts.rhmoxide0.MnO*79.8768/70.9374)*71.8444/79.8768)*(1+159.6882/71.8444)/100;
                Magnetite(Magnetite<0)=0;
                Hematite=(melts.rhmoxide0.Fe2O3-Magnetite*100*159.6882/231.5326)/100;
                melts.(['ilmenite' melts.minerals{i}(9:end)]).mass=melts.(melts.minerals{i}).mass.*Ilmenite;
                melts.(['ilmenite' melts.minerals{i}(9:end)]).Temperature=melts.(melts.minerals{i}).Temperature;
                melts.(['magnetite' melts.minerals{i}(9:end)]).mass=melts.(melts.minerals{i}).mass.*Magnetite;
                melts.(['magnetite' melts.minerals{i}(9:end)]).Temperature=melts.(melts.minerals{i}).Temperature;
                melts.(['hematite' melts.minerals{i}(9:end)]).mass=melts.(melts.minerals{i}).mass.*Hematite;
                melts.(['hematite' melts.minerals{i}(9:end)]).Temperature=melts.(melts.minerals{i}).Temperature;
                melts.minerals=unique([melts.minerals; ['ilmenite' melts.minerals{i}(9:end)]; ['magnetite' melts.minerals{i}(9:end)]; ['hematite' melts.minerals{i}(9:end)];],'stable');
            end
        end
        
        
        
        % Make a new struct containing zero-extended mineral masses
        mass=struct;
        mass.minerals=melts.minerals;
        mass.solids=zeros(simlength,1);
        for i=1:length(melts.minerals)
            melts.(melts.minerals{i}).Index=round((maxT-melts.(melts.minerals{i}).Temperature)./deltaT)+1;
            mass.(melts.minerals{i})=zeros(simlength,1);
            mass.(melts.minerals{i})(melts.(melts.minerals{i}).Index)=melts.(melts.minerals{i}).mass;
            if i>1 && isempty(strfind(melts.minerals{i},'feldspar')) && isempty(strfind(melts.minerals{i},'rhmoxide'))
                mass.solids=mass.solids+mass.(melts.minerals{i});
            end
        end
        mass.minerals=melts.minerals;
        
        
        % Fill in mineral compositions
        for i=1:length(melts.minerals)
            if any(ismember({'quartz','apatite','whitlockite','aenigmatite'},melts.minerals{i}))
                for e=elements
                    melts.(melts.minerals{i}).(e{:})=0;
                    if isfield(ref.(melts.minerals{i}),e)
                        melts.(melts.minerals{i}).(e{:})=ref.(melts.minerals{i}).(e{:});
                    end
                end
            end
        end
        
        
        % Calculate solid major element composition
        s=struct;
        for j=1:length(elements)
            s.(elements{j})=NaN(simlength,1);
            for i=2:length(melts.minerals)
                if isfield(melts.(melts.minerals{i}),elements{j})
                    s.(elements{j})(melts.(melts.minerals{i}).Index) = nansum([s.(elements{j})(melts.(melts.minerals{i}).Index) ...
                        melts.(melts.minerals{i}).(elements{j}) .* melts.(melts.minerals{i}).mass], 2);
                end
                
            end
            s.(elements{j}) = s.(elements{j})./mass.solids;
        end
        
        
        
        % Calculate trace element compositions
        meltSi=round(melts.liquid_0.SiO2);
        meltSi(meltSi<40)=40; meltSi(meltSi>80)=80;
        
        % Calculate where SiO2 is increasing during differentiation, so as to plot only that
        posSi=melts.liquid_0.SiO2>([0; melts.liquid_0.SiO2(1:end-1)]-0.01);
        
        for e=1:length(traceelements);
            % Calculate bulk partition coeff.
            D.(traceelements{e})=zeros(simlength,1);
            for i=1:length(p.minerals)
                mnrlindxs=find(~cellfun('isempty',strfind(melts.minerals, lower(p.minerals{i}))));
                for k=1:length(mnrlindxs)
                    % Note that minerals that we don't have data for end up being
                    % treated like all elements are incompatible in them.
                    % Note, geometric mean = log average
                    D.(traceelements{e})=nansum([D.(traceelements{e}) mass.(mass.minerals{mnrlindxs(k)}) .* (10.^p.(p.minerals{i}).(traceelements{e})(meltSi-39)) ./mass.solids],2);
                end
            end
            
            % Calculate amount of mass removed as solid in each step
            masssolidFract=([100; mass.liquid_0(1:end-1)]-mass.liquid_0)/100;
            solidFract=masssolidFract./[100; mass.liquid_0(1:end-1)]*100;
            
            % Calculate liquid composition
            l.(traceelements{e})=NaN(simlength+1,1);
            l.(traceelements{e})(1)=tr(e, startingcol);
            % Smooth fractional crystallization
            for i=1:simlength
                l.(traceelements{e})(i+1)=l.(traceelements{e})(i)*(1-solidFract(i)).^(D.(traceelements{e})(i)-1);
            end
            
            % Calculate solid composition
            % Mass of element in liquid
            mEl=l.(traceelements{e}).*[100; mass.liquid_0]/100;
            % Mass of trace element lost at each step
            dmEl=mEl(1:end-1)-mEl(2:end);
            % Concentration in solid
            s.(traceelements{e})=dmEl./masssolidFract;
            
            %%%%%%%%%%%%%%%%%%%%%%%% Plot results %%%%%%%%%%%%%%%%%%%%%%%%%
            plotwater=(nanmean(melts.liquid_0.H2O)-0)/4; %Scale for 0-4% H2O
            if plotwater>1; plotwater=1; end
            if plotwater<0; plotwater=0; end
            linecolor=cmap(ceil(plotwater*100),:);
            
            hold on; subaxis(yfigs,xfigs,mod(e-1,xfigs*yfigs)+1);
            % melt
            plot(melts.liquid_0.SiO2(posSi),l.(traceelements{e})(logical([posSi; 0])),'Color',linecolor)
            % cumulate (color each point individually by melt silica)
            for step=1:length(s.SiO2)
                if posSi(step)&&mass.solids(step)>0
                    plot(s.SiO2(step),s.(traceelements{e})(step),'.','MarkerSize',mass.solids(step),'Color',[0 (melts.liquid_0.SiO2(step)-45)/45 0]);
                end
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end
    end
end


% Re-plot v-p trends on top
for i=1:length(traceelements)
    hold on; subaxis(yfigs,xfigs,mod(i-1,xfigs*yfigs)+1); binplot(mcplutonic.SiO2,mcplutonic.(traceelements{i}),40,80,length(mcplutonic.SiO2)./length(plutonic.SiO2),20,'.b')
    hold on; subaxis(yfigs,xfigs,mod(i-1,xfigs*yfigs)+1); binplot(mcvolcanic.SiO2,mcvolcanic.(traceelements{i}),40,80,length(mcvolcanic.SiO2)./length(volcanic.SiO2),20,'.r')
    xlim([30 80])
    colormap([0.4*ones(54,2),(0.47:0.01:1)'])
    caxis([1 3])
end
