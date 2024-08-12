
from abaqus import mdb
from abaqus import *
from abaqusConstants import *
from caeModules import *
from part import *
from database import *
from database_output import *

def assemble(model_set, pressure, cf3_values):

    # Access the model database and root assembly
    myModel = mdb.models[model_set]
    myAssembly = myModel.rootAssembly

    # Create a material property for the bone
    myMaterialBone = myModel.Material(name='BONES')
    myMaterialBone.Elastic(table=((7300.0, 0.3),))

    # Create a material property for the tissue
    myMaterialTissue = myModel.Material(name='SKIN')
    myMaterialTissue.Elastic(table=((1.15, 0.49),))

    # Create a section for the bone using the bone material property
    mySectionBone = myModel.HomogeneousSolidSection(name='BONES', material='BONES')

    # Create a section for the tissue using the tissue material property
    mySectionTissue = myModel.HomogeneousSolidSection(name='SKIN', material='SKIN')


    # Assign the bone section to the bone part
    myPartBone = myModel.parts['BONES']
    # create a set named 'bones' for all the nodes in the part
    set_bones = myPartBone.Set(elements=myPartBone.elements, name='BONES')
    myPartBone.SectionAssignment(sectionName='BONES', region=set_bones)

    # Assign the tissue section to the tissue part
    myPartTissue = myModel.parts['SKIN']
    # create a set named 'tissue' for all the nodes in the part
    set_tissue = myPartTissue.Set(elements=myPartTissue.elements, name='SKIN')
    myPartTissue.SectionAssignment(sectionName='SKIN', region=set_tissue)



    # Create an instance of the bone part in the assembly
    myPartBone = myModel.parts['BONES']
    myAssembly.Instance(name='BONE-1', part=myPartBone, dependent=OFF)

    # Create an instance of the tissue part in the assembly
    myPartTissue = myModel.parts['SKIN']
    myAssembly.Instance(name='SKIN-1', part=myPartTissue, dependent=OFF)

    # Locate the nodes for the start and end points
    # startNode = myAssembly.instances['BONE-1'].nodes[214]
    # endNodes = [myAssembly.instances['BONE-1'].nodes[i] for i in [5453, 5640, 5618, 5347, 5048]]
    startNode = myAssembly.instances['BONE-1'].nodes[508]
    endNodes = [myAssembly.instances['BONE-1'].nodes[i] for i in [5529, 5508, 5444, 5262, 4931]]
    # Create float objects for the behavior options

    
    myConnectorSection = myModel.ConnectorSection(
        name='PF', 
        assembledType=SLIPRING, massPerLength=1000)
    myModel.sections['PF'].setValues(contactAngle=0.0)

    # Create the connectors using the start and end nodes
    for i, endNode in enumerate(endNodes, start=1):

        edge = myAssembly.WirePolyLine(points=((startNode, endNode), ), #mergeWire=OFF,
                                meshable=OFF)
        myAssembly.Set(name='Wire' + str(i), edges=myAssembly.getFeatureEdges(edge.name))
        csa = myAssembly.SectionAssignment(region=myAssembly.sets['Wire' + str(i)],
                                    sectionName='PF')
        myAssembly.ConnectorOrientation(region=csa.getSet())
    
    # create a constrain between bone and tissue
    
    # create steps based on previous steps
    previous_step = 'Initial'
    for i in ['hindfoot','midfoot','forefoot']:
        step = myModel.StaticStep(name='Step_'+i, previous=previous_step, 
                        nlgeom=ON, description='',
                        timePeriod=1.0, timeIncrementationMethod=AUTOMATIC,
                        maxNumInc=100, initialInc=1, minInc=1e-05, maxInc=1
                        )
        previous_step = 'Step_'+i


    f1 = myAssembly.instances['BONE-1'].elements
    face1Elements1 = f1.getSequenceFromMask(mask=boneface1)
    face2Elements1 = f1.getSequenceFromMask(mask=boneface2)
    face3Elements1 = f1.getSequenceFromMask(mask=boneface3)
    face4Elements1 = f1.getSequenceFromMask(mask=boneface4)
    myAssembly.Surface(face1Elements=face1Elements1, face2Elements=face2Elements1, 
        face3Elements=face3Elements1, face4Elements=face4Elements1, name='Surf-bones')
    #: The surface 'Surf-5' has been created (14214 mesh faces).

    f1 = myAssembly.instances['SKIN-1'].elements
    face1Elements1 = f1.getSequenceFromMask(mask = skinface1)
    face2Elements1 = f1.getSequenceFromMask(mask = skinface2)
    face3Elements1 = f1.getSequenceFromMask(mask = skinface3)
    face4Elements1 = f1.getSequenceFromMask(mask = skinface4)
    myAssembly.Surface(face1Elements=face1Elements1, face2Elements=face2Elements1, 
        face3Elements=face3Elements1, face4Elements=face4Elements1, 
        name='Surf-skin')

    # Create two sets of surfaces to be tied
    set1 = myAssembly.surfaces['Surf-bones']
    set2 = myAssembly.surfaces['Surf-skin']
    # Create a tie constraint between the two sets of nodes
    tie_name = 'Constraint-tie'
    myModel.Tie(name=tie_name, main=set1, secondary=set2, positionToleranceMethod=COMPUTED, adjust=ON, 
    tieRotations=ON, thickness=ON)


    n1 = myAssembly.instances['SKIN-1'].nodes
    nodes1 = n1.getSequenceFromMask(mask = BC_face)
    myAssembly.Set(nodes=nodes1, name='Set_BC')



    f1 = myAssembly.instances['SKIN-1'].elements
    face1Elements1 = f1.getSequenceFromMask(mask = hindfootface1)
    face2Elements1 = f1.getSequenceFromMask(mask = hindfootface2)
    face3Elements1 = f1.getSequenceFromMask(mask = hindfootface3)
    myAssembly.Surface(face1Elements=face1Elements1, face2Elements=face2Elements1, 
        face3Elements=face3Elements1, name='Surf-hindfoot')
    #: The surface 'Surf-1' has been created (954 mesh faces).


    f1 = myAssembly.instances['SKIN-1'].elements
    face1Elements1 = f1.getSequenceFromMask(mask = midfootface1)
    face3Elements1 = f1.getSequenceFromMask(mask = midfootface2)
    face4Elements1 = f1.getSequenceFromMask(mask = midfootface3)
    myAssembly.Surface(face1Elements=face1Elements1, face3Elements=face3Elements1, 
        face4Elements=face4Elements1, name='Surf-midfoot')
    #: The surface 'Surf-2' has been created (682 mesh faces).


    f1 = myAssembly.instances['SKIN-1'].elements
    face1Elements1 = f1.getSequenceFromMask(mask = forefootface1)
    face2Elements1 = f1.getSequenceFromMask(mask = forefootface2)
    face3Elements1 = f1.getSequenceFromMask(mask = forefootface3)
    myAssembly.Surface(face1Elements=face1Elements1, face2Elements=face2Elements1, 
        face3Elements=face3Elements1, name='Surf-forefoot')
    #: The surface 'Surf-3' has been created (1579 mesh faces).


    f1 = myAssembly.instances['SKIN-1'].elements
    face1Elements1 = f1.getSequenceFromMask(mask = toesface)
    myAssembly.Surface(face1Elements=face1Elements1, name='Surf-toes')
    #: The surface 'Surf-4' has been edited (752 mesh faces).


    ### this section create five ATs ###
    # Create a reference point at the specific nodes
    n1 = myAssembly.instances['BONE-1'].nodes
    r11 = myAssembly.referencePoints
    node_ids = [178, 136, 129, 126, 137]
    points = [(37,28.35256, 104.285698), (37,24.27084, 104.285698), (37,21.367279, 104.285698), (37,18.88236, 104.285698), (37,13.72351, 104.285698)]
    myModel.ConnectorSection(name='ConnSect-AT', translationalType=AXIAL, rotationalType=NONE)
   

    for i in range(len(points)):
        r1 = myAssembly.ReferencePoint(point=points[i])
        myAssembly.WirePolyLine(points=((n1[node_ids[i]], r11[r1.id]), ), mergeType=IMPRINT, meshable=OFF)
        e1 = myAssembly.edges
        edges1 = e1.getSequenceFromMask(mask=('[#1 ]', ), )
        Line_AT_name = 'Wire-AT-Set-' + str(i+1)
        myAssembly.Set(edges=edges1, name=Line_AT_name)
        Line_AT = myAssembly.sets[Line_AT_name]
        csa = myAssembly.SectionAssignment(sectionName='ConnSect-AT', region=Line_AT)
        

    ### setting boundary conditions ###
    # Define two sets of reference points
    region1 = myAssembly.sets['Set_BC']

    nodes_bone_BC = n1.getSequenceFromMask(mask=(
    '[#0:25 #80000000 #0:6 #41 #0:3 #200000 #2002', 
    ' #0 #200000 #4028 #8800005 #20120080 #40808 #c280', 
    ' #c100001 #300c0 #80000013 #4028030 #28058008 #5000000 #20814002', 
    ' #202 #2142041 #44001000 #10084 #c805200 #5000 #48185000', 
    ' #2000000 #4a90000 #10000 #2000 #240900 #8000040 #10280', 
    ' #2200000 #2040040 #22080 #10000000 #1 #44000110 #80', 
    ' #2 #12080000 #8200 ]', ), )
    region2 = myAssembly.Set(nodes=nodes_bone_BC, name='Set-BC-Bone')
    
    refPoints = []
    r1 = myAssembly.referencePoints
    for i in r1.keys():
        refPoints.append(r1[i])
    refPoints = tuple(refPoints)
    region3 = myAssembly.Set(referencePoints=refPoints, name='Set-AT-BC') 


    step_names = ['Step_hindfoot','Step_midfoot','Step_forefoot']
    bc_names1 = ['BC-hindfoot','BC-midfoot','BC-forefoot']
    bc_names2 = ['BC-bone-hindfoot','BC-bone-midfoot','BC-bone-forefoot']
    bc_names3 = ['BC-at-hindfoot','BC-at-midfoot','BC-at-forefoot']

    # Create two BCs for each step
    for i in range(len(step_names)):
        myModel.DisplacementBC(name=bc_names1[i], 
            createStepName=step_names[i], region=region1, u1=0.0, u2=0.0, u3=0.0, 
            ur1=UNSET, ur2=UNSET, ur3=UNSET, amplitude=UNSET, fixed=OFF, 
            distributionType=UNIFORM, fieldName='', localCsys=None)

        myModel.DisplacementBC(name=bc_names2[i], 
            createStepName=step_names[i], region=region2, u1=0.0, u2=0.0, u3=0.0, 
            ur1=UNSET, ur2=UNSET, ur3=UNSET, amplitude=UNSET, fixed=OFF, 
            distributionType=UNIFORM, fieldName='', localCsys=None)
        myModel.DisplacementBC(name=bc_names3[i], 
            createStepName=step_names[i], region=region3, u1=0.0, u2=0.0, u3=0.0, 
            ur1=UNSET, ur2=UNSET, ur3=UNSET, amplitude=UNSET, fixed=OFF, 
            distributionType=UNIFORM, fieldName='', localCsys=None)
        


    # Define the names of the loads, steps, and surfaces
    load_names =    ['Load-hindfoot', 'Load-midfoot-1', 'Load-midfoot-2', 'Load-midfoot-3', 'Load-midfoot-4', 'Load-forefoot-1','Load-forefoot-2']
    step_names =    ['Step_hindfoot', 'Step_midfoot','Step_midfoot', 'Step_midfoot', 'Step_midfoot', 'Step_forefoot','Step_forefoot']
    surface_names = ['Surf-hindfoot', 'Surf-toes', 'Surf-forefoot','Surf-midfoot', 'Surf-hindfoot', 'Surf-toes', 'Surf-forefoot']
    
    # Define a set to keep track of which step names have already been processed
    processed_steps = set()


    # Loop through the names and create corresponding pressure loads
    for _, (load_name, step_name, surface_name, magnitude) in enumerate(zip(load_names, step_names, surface_names, pressure)):
        region = myAssembly.surfaces[surface_name]
        print('This is ' + str(magnitude))
        myModel.Pressure(name=load_name,
                        createStepName=step_name,
                        region=region,
                        distributionType=UNIFORM,
                        field='',
                        magnitude=magnitude,
                        amplitude=UNSET)
        if step_name in ['Step_midfoot', 'Step_forefoot'] and step_name not in processed_steps:
            for j in range(1, 6): # To apply connector force to the 5 wires
                Connector_force_name = 'Load-AT_{}_{}'.format(step_name, j)
                Line_AT_name = 'Wire-AT-Set-' + str(j)
                Line_AT = myAssembly.sets[Line_AT_name]
                myModel.ConnectorForce(name=Connector_force_name, 
                    createStepName=step_name, region=Line_AT, f1=-cf3_values[step_name]/5)
            # Add the step name to the processed set
            processed_steps.add(step_name)



    myModel.FieldOutputRequest(name='F-Output-2', 
        createStepName='Step_midfoot', variables=('S', 'MISES', 'MISESMAX', 'E', 'PE',
        'PEEQ', 'U', 'RF', 'CF', 'CSTRESS', 'CDISP'))

    myModel.FieldOutputRequest(name='F-Output-3', 
        createStepName='Step_hindfoot', variables=('S', 'MISES', 'MISESMAX', 'E', 'PE',
        'PEEQ', 'U', 'RF', 'CF', 'CSTRESS', 'CDISP'))

    list_output = [Set_M1, Set_M2, Set_M3, Set_M4, Set_M5, Set_Calcaneus, Set_Talus]
    name = ['set_M1', 'set_M2', 'set_M3', 'set_M4', 'set_M5', 'set_Calcaneus', 'set_Talus']        
    # create a set of element to extract results
    e1 = myAssembly.instances['BONE-1'].elements
    for i, j in zip(list_output, name):
        elements1 = e1.getSequenceFromMask(mask = i)
        myAssembly.Set(elements=elements1, name= j)
