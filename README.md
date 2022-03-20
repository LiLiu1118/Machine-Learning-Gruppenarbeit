# Machine-Learning-Gruppenarbeit
# Performance comparison – regional differences in the wind farm

## Background 
Although the basically available land potential for the installation of wind farms in Germany amounts almost 14% of the national area, many restrictions have to be made. Distance to housing estates defined by law, missing acceptance of wind farms by citizens and site owners and area development objectives of local authorities  leads  to  a  drastically  reduced  available  space  for  wind  farms.  Therefore,  and  naturally  for  an  increased  economic  competitiveness,  the potential exploitation of wind farms is a top objective. Case studies of existing wind farms have shown substantial losses of performance due to various reasons (e.g. wake  effects,  regional  wind  speed  variations  etc.).  Characterizing  the  regional differences helps   identifying   performance   potentials   as   well   as   ensuring   the   stability   of   the   energy   network. Moreover, the  generated   knowledge can then be used to develop more efficient wind farms. Problem DefinitionConduct a performance comparison where you identify regional differences in each  wind  farm  individually  (consider  the  wind  farms  separately  from  each  other). Investigate on bundling wind turbines within the wind farm in different performance clusters. Choose a reasonable methodology to develop a schematic regional map and investigate on the reasoning of efficiency differences between wind turbines within the wind farm. Examine on seasonal differences effecting the performance history.

## Main objective: 

Identify regional differences in each wind farm within a performance comparison (consider the wind farms separatelyfrom each other)

Identify meaningful physical parameters 

Analyze the existing database 

Bundle the wind turbines in different performance clusters

Develop a schematic regional map

Investigate on efficiency differences between the wind turbines

Examine on seasonal differences

## Results

<p float="left">
  <img src="https://github.com/LiLiu1118/Machine-Learning-Gruppenarbeit/blob/main/Abbildungen/WindRosePark1.png" width="300" height="200"/>
  <img src="https://github.com/LiLiu1118/Machine-Learning-Gruppenarbeit/blob/main/Abbildungen/WindRosePark2.png"  width="300" height="200" /> 
</p>

<p float="left">
  <img src="https://github.com/LiLiu1118/Machine-Learning-Gruppenarbeit/blob/main/Abbildungen/SOM-Park1.png" width="300" height="200"/>
  <img src="https://github.com/LiLiu1118/Machine-Learning-Gruppenarbeit/blob/main/Abbildungen/AmbientHirachicalPark1.png" width="300" height="200"/>
</p>


From the given wind farms it can be seen that the power output can be concluded with relatively good accuracy using hierarchical clustering based on the environmental parameters. However, the turbines are regulated from a certain power level upwards, so that high wind speeds no longer lead to a better output. The self-organizing maps made it clear that the operating company should pay very close attention to the wind speed. Ideally, positions should be chosen which do not necessarily have very high wind speeds, but a steady current. After all, this is by far the most important factor for increasing electricity production and in this case the energy can be used best. 

It is also clear that the ambient temperature does not have as great an influence on the energy output as one might think. The different densities do not result in significantly higher values, but a correlation can occur due to the different seasons and the different wind percentages that occur in them.

For more details, please check out [Bericht.pdf](https://github.com/LiLiu1118/Machine-Learning-Gruppenarbeit/blob/main/Bericht.pdf).
