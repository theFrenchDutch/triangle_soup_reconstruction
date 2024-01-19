using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using Unity.Mathematics;


[CustomEditor(typeof(MutationOptimizer))]
public class MutationOptimizerEditor : Editor
{
	SerializedProperty targetResolution;
	SerializedProperty targetMode;
	SerializedProperty target2DImage;
	SerializedProperty target3DMesh;
	SerializedProperty target3DCOLMAPFolder;
	SerializedProperty targetMaterial;
	SerializedProperty colmapRescaler;
	SerializedProperty colmapUseMasking;
	SerializedProperty optimPrimitive;
	SerializedProperty primitiveCount;
	SerializedProperty primitiveInitSize;
	SerializedProperty primitiveInitSeed;
	SerializedProperty voxelGridMaxResolution;
	SerializedProperty materialResolution;
	SerializedProperty meshInitState;
	SerializedProperty meshCatmullClarkSubdivisions;
	SerializedProperty meshCatmullClarkUseCreases;
	SerializedProperty meshInitSize;
	SerializedProperty initPrimitivesOnMeshSurface;
	SerializedProperty colmapUseCameraBounds;
	SerializedProperty colmapOnePrimitivePerInitPoint;
	SerializedProperty colmapUseInitPointDistances;
	SerializedProperty colmapMaxInitPointDistance;
	SerializedProperty colmapInitPointsTrimAABBCenterSize;
	SerializedProperty sphericalHarmonicsMode;
	SerializedProperty transparencyMode;
	SerializedProperty maxFragmentsPerPixel;
	SerializedProperty alphaContributingCutoff;
	SerializedProperty materialHeightMode;
	SerializedProperty targetTessellationResolution;
	SerializedProperty randomViewZoomRange;

	SerializedProperty reset;
	SerializedProperty pause;
	SerializedProperty stepForward;
	SerializedProperty displayMode;
	SerializedProperty separateFreeViewCamera;
	SerializedProperty displayCatmullClarkControlCage;

	SerializedProperty lrGlobalStartMulAndSpeed;
	SerializedProperty lrGeometryStartMulAndSpeed;
	SerializedProperty lrRegulStartMulAndSpeed;
	SerializedProperty doublingAmount;
	SerializedProperty doubleEveryXSteps;

	SerializedProperty optimizer;
	SerializedProperty gradientsWarmupSteps;
	SerializedProperty optimSupersampling;
	SerializedProperty doAlphaLoss;
	SerializedProperty doAllInputFramesForEachOptimStep;
	SerializedProperty viewsPerOptimStep;
	SerializedProperty antitheticMutationsPerFrame;
	SerializedProperty parameterSeparationMode;

	SerializedProperty globalLearningRate;
	SerializedProperty beta1;
	SerializedProperty beta2;
	SerializedProperty learningRatePosition;
	SerializedProperty learningRateCrease;
	SerializedProperty learningRateUV;
	SerializedProperty learningRateRotation;
	SerializedProperty learningRateScale;
	SerializedProperty learningRateColor;
	SerializedProperty learningRateAlpha;
	SerializedProperty learningRateNormal;
	SerializedProperty learningRateMetallic;
	SerializedProperty learningRateRoughness;
	SerializedProperty learningRateHeight;
	SerializedProperty vertexRegularizerWeight;
	SerializedProperty vertexRegularizer2Weight;

	SerializedProperty doPrimitiveResampling;
	SerializedProperty resamplingInterval;
	SerializedProperty optimStepsUnseenBeforeKill;
	SerializedProperty minPrimitiveWorldArea;

	SerializedProperty currentOptimStep;
	SerializedProperty totalElapsedSeconds;
	SerializedProperty millisecondsPerOptimStep;
	SerializedProperty internalOptimResolution;
	SerializedProperty voxelGridResolution;
	SerializedProperty colmapImageCount;

	SerializedProperty needsToDoublePrimitives;
	SerializedProperty needsToDoubleVolumeRes;
	SerializedProperty needsToCustomExport;

	void OnEnable()
	{
		targetResolution = serializedObject.FindProperty("targetResolution");
		targetMode = serializedObject.FindProperty("targetMode");
		target2DImage = serializedObject.FindProperty("target2DImage");
		target3DMesh = serializedObject.FindProperty("target3DMesh");
		target3DCOLMAPFolder = serializedObject.FindProperty("target3DCOLMAPFolder");
		targetMaterial = serializedObject.FindProperty("targetMaterial");
		colmapRescaler = serializedObject.FindProperty("colmapRescaler");
		colmapUseMasking = serializedObject.FindProperty("colmapUseMasking");
		optimPrimitive = serializedObject.FindProperty("optimPrimitive");
		primitiveCount = serializedObject.FindProperty("primitiveCount");
		primitiveInitSize = serializedObject.FindProperty("primitiveInitSize");
		primitiveInitSeed = serializedObject.FindProperty("primitiveInitSeed");
		voxelGridMaxResolution = serializedObject.FindProperty("voxelGridMaxResolution");
		materialResolution = serializedObject.FindProperty("materialResolution");
		meshInitState = serializedObject.FindProperty("meshInitState");
		meshInitSize = serializedObject.FindProperty("meshInitSize");
		meshCatmullClarkSubdivisions = serializedObject.FindProperty("meshCatmullClarkSubdivisions");
		meshCatmullClarkUseCreases = serializedObject.FindProperty("meshCatmullClarkUseCreases");
		initPrimitivesOnMeshSurface = serializedObject.FindProperty("initPrimitivesOnMeshSurface");
		colmapUseCameraBounds = serializedObject.FindProperty("colmapUseCameraBounds");
		colmapOnePrimitivePerInitPoint = serializedObject.FindProperty("colmapOnePrimitivePerInitPoint");
		colmapUseInitPointDistances = serializedObject.FindProperty("colmapUseInitPointDistances");
		colmapMaxInitPointDistance = serializedObject.FindProperty("colmapMaxInitPointDistance");
		colmapInitPointsTrimAABBCenterSize = serializedObject.FindProperty("colmapInitPointsTrimAABBCenterSize");
		sphericalHarmonicsMode = serializedObject.FindProperty("sphericalHarmonicsMode");
		transparencyMode = serializedObject.FindProperty("transparencyMode");
		maxFragmentsPerPixel = serializedObject.FindProperty("maxFragmentsPerPixel");
		alphaContributingCutoff = serializedObject.FindProperty("alphaContributingCutoff");
		materialHeightMode = serializedObject.FindProperty("materialHeightMode");
		targetTessellationResolution = serializedObject.FindProperty("targetTessellationResolution");
		randomViewZoomRange = serializedObject.FindProperty("randomViewZoomRange");

		reset = serializedObject.FindProperty("reset");
		pause = serializedObject.FindProperty("pause");
		stepForward = serializedObject.FindProperty("stepForward");
		displayMode = serializedObject.FindProperty("displayMode");
		separateFreeViewCamera = serializedObject.FindProperty("separateFreeViewCamera");
		displayCatmullClarkControlCage = serializedObject.FindProperty("displayCatmullClarkControlCage");

		lrGlobalStartMulAndSpeed = serializedObject.FindProperty("lrGlobalStartMulAndSpeed");
		lrGeometryStartMulAndSpeed = serializedObject.FindProperty("lrGeometryStartMulAndSpeed");
		lrRegulStartMulAndSpeed = serializedObject.FindProperty("lrRegulStartMulAndSpeed");
		doublingAmount = serializedObject.FindProperty("doublingAmount");
		doubleEveryXSteps = serializedObject.FindProperty("doubleEveryXSteps");

		optimizer = serializedObject.FindProperty("optimizer");
		gradientsWarmupSteps = serializedObject.FindProperty("gradientsWarmupSteps");
		optimSupersampling = serializedObject.FindProperty("optimSupersampling");
		doAlphaLoss = serializedObject.FindProperty("doAlphaLoss");
		doAllInputFramesForEachOptimStep = serializedObject.FindProperty("doAllInputFramesForEachOptimStep");
		viewsPerOptimStep = serializedObject.FindProperty("viewsPerOptimStep");
		antitheticMutationsPerFrame = serializedObject.FindProperty("antitheticMutationsPerFrame");
		parameterSeparationMode = serializedObject.FindProperty("parameterSeparationMode");

		globalLearningRate = serializedObject.FindProperty("globalLearningRate");
		beta1 = serializedObject.FindProperty("beta1");
		beta2 = serializedObject.FindProperty("beta2");
		learningRatePosition = serializedObject.FindProperty("learningRatePosition");
		learningRateCrease = serializedObject.FindProperty("learningRateCrease");
		learningRateUV = serializedObject.FindProperty("learningRateUV");
		learningRateRotation = serializedObject.FindProperty("learningRateRotation");
		learningRateScale = serializedObject.FindProperty("learningRateScale");
		learningRateColor = serializedObject.FindProperty("learningRateColor");
		learningRateAlpha = serializedObject.FindProperty("learningRateAlpha");
		learningRateNormal = serializedObject.FindProperty("learningRateNormal");
		learningRateMetallic = serializedObject.FindProperty("learningRateMetallic");
		learningRateRoughness = serializedObject.FindProperty("learningRateRoughness");
		learningRateHeight = serializedObject.FindProperty("learningRateHeight");
		vertexRegularizerWeight = serializedObject.FindProperty("vertexRegularizerWeight");
		vertexRegularizer2Weight = serializedObject.FindProperty("vertexRegularizer2Weight");

		doPrimitiveResampling = serializedObject.FindProperty("doPrimitiveResampling");
		resamplingInterval = serializedObject.FindProperty("resamplingInterval");
		optimStepsUnseenBeforeKill = serializedObject.FindProperty("optimStepsUnseenBeforeKill");
		minPrimitiveWorldArea = serializedObject.FindProperty("minPrimitiveWorldArea");

		currentOptimStep = serializedObject.FindProperty("currentOptimStep");
		totalElapsedSeconds = serializedObject.FindProperty("totalElapsedSeconds");
		millisecondsPerOptimStep = serializedObject.FindProperty("millisecondsPerOptimStep");
		internalOptimResolution = serializedObject.FindProperty("internalOptimResolution");
		voxelGridResolution = serializedObject.FindProperty("voxelGridResolution");
		colmapImageCount = serializedObject.FindProperty("colmapImageCount");

		needsToDoublePrimitives = serializedObject.FindProperty("needsToDoublePrimitives");
		needsToDoubleVolumeRes = serializedObject.FindProperty("needsToDoubleVolumeRes");
		needsToCustomExport = serializedObject.FindProperty("needsToCustomExport");
	}

	public void ValidateParameters()
	{
		// Validate parameters
		if (doPrimitiveResampling.boolValue == true)
			primitiveCount.intValue = math.max(primitiveCount.intValue, 2);
		optimSupersampling.floatValue = math.max(optimSupersampling.floatValue, 1.0f / 8.0f);
		viewsPerOptimStep.intValue = math.max(viewsPerOptimStep.intValue, 1);
		targetResolution.vector2IntValue = new Vector2Int(math.max(targetResolution.vector2IntValue.x, 2), math.max(targetResolution.vector2IntValue.y, 2));
		voxelGridMaxResolution.intValue = math.max(voxelGridMaxResolution.intValue, 2);
		maxFragmentsPerPixel.intValue = math.max(maxFragmentsPerPixel.intValue, 1);
		maxFragmentsPerPixel.intValue = math.min(maxFragmentsPerPixel.intValue, 64);
		meshCatmullClarkSubdivisions.intValue = math.max(meshCatmullClarkSubdivisions.intValue, 0);
		if (doAllInputFramesForEachOptimStep.boolValue == true && targetMode.enumValueIndex == 2 && colmapImageCount.intValue > 0)
			viewsPerOptimStep.intValue = colmapImageCount.intValue;
		if (optimPrimitive.enumValueIndex != 0 && optimPrimitive.enumValueIndex != 1 && optimPrimitive.enumValueIndex != 2 && optimPrimitive.enumValueIndex != 4)
			doPrimitiveResampling.boolValue = false;
		//if (targetMode.enumValueIndex != 0)
		//	framesUnseenBeforeKill.intValue = math.max(framesUnseenBeforeKill.intValue, framesPerOptimStep.intValue * antitheticMutationsPerFrame.intValue * 2);
		if (optimPrimitive.enumValueIndex == 3)
		{
			transparencyMode.enumValueIndex = 0;
			if (targetMode.enumValueIndex == 0)
				targetMode.enumValueIndex = 1;
		}
		if (optimPrimitive.enumValueIndex == 5 || optimPrimitive.enumValueIndex == 6)
			transparencyMode.enumValueIndex = 0;
		if ((optimPrimitive.enumValueIndex == 4 || optimPrimitive.enumValueIndex == 2) && transparencyMode.enumValueIndex == 0)
			transparencyMode.enumValueIndex = 1;
		if (optimPrimitive.enumValueIndex == 5)
			doPrimitiveResampling.boolValue = false;
		if (targetMode.enumValueIndex == 3)
			optimPrimitive.enumValueIndex = 5;
		if (optimPrimitive.enumValueIndex == 6 && parameterSeparationMode.enumValueIndex == 0)
		{
			parameterSeparationMode.enumValueIndex = 1;
		}

		if (optimPrimitive.enumValueIndex == 5)
		{
			meshCatmullClarkSubdivisions.intValue = math.max(meshCatmullClarkSubdivisions.intValue, 1);
		}
		if (optimPrimitive.enumValueIndex == 5 && materialHeightMode.enumValueIndex != 2)
		{
			meshCatmullClarkSubdivisions.intValue = 1;
		}

		if (meshCatmullClarkUseCreases.boolValue == false)
			learningRateCrease.floatValue = 0.0f;

		if (optimPrimitive.enumValueIndex == 5 || optimPrimitive.enumValueIndex == 6 || targetMode.enumValueIndex == 3)
			sphericalHarmonicsMode.enumValueIndex = 0;

		if (targetMode.enumValueIndex == 0)
			viewsPerOptimStep.intValue = 1;
	}

	public override void OnInspectorGUI()
	{
		serializedObject.Update();

		// Metrics display
		EditorGUILayout.LabelField("Metrics Display", EditorStyles.boldLabel);
		EditorGUI.BeginDisabledGroup(true);
		EditorGUILayout.PropertyField(currentOptimStep);
		EditorGUILayout.PropertyField(totalElapsedSeconds);
		EditorGUILayout.PropertyField(millisecondsPerOptimStep);
		EditorGUILayout.PropertyField(internalOptimResolution);
		if (optimPrimitive.enumValueIndex == 3)
			EditorGUILayout.PropertyField(voxelGridResolution);
		if (targetMode.enumValueIndex == 2)
			EditorGUILayout.PropertyField(colmapImageCount);
		EditorGUI.EndDisabledGroup();
		EditorGUILayout.Space();

		// Scene settings
		EditorGUILayout.LabelField("Scene Settings", EditorStyles.boldLabel);
		if (targetMode.enumValueIndex != 2)
			EditorGUILayout.PropertyField(targetResolution);
		EditorGUILayout.PropertyField(targetMode);
		if (targetMode.enumValueIndex == 0)
			EditorGUILayout.PropertyField(target2DImage);
		else if (targetMode.enumValueIndex == 1)
			EditorGUILayout.PropertyField(target3DMesh);
		else if (targetMode.enumValueIndex == 2)
		{
			EditorGUILayout.PropertyField(target3DCOLMAPFolder);
			EditorGUILayout.PropertyField(colmapRescaler);
			EditorGUILayout.PropertyField(colmapUseMasking);
		}
		else if (targetMode.enumValueIndex == 3)
		{
			EditorGUILayout.PropertyField(targetMaterial);
			EditorGUILayout.PropertyField(targetTessellationResolution);
		}
		EditorGUILayout.PropertyField(optimPrimitive);
		EditorGUILayout.PropertyField(primitiveInitSeed);
		if (optimPrimitive.enumValueIndex == 0 || optimPrimitive.enumValueIndex == 1 || optimPrimitive.enumValueIndex == 2 || optimPrimitive.enumValueIndex == 4)
		{
			EditorGUILayout.PropertyField(primitiveCount);
			EditorGUILayout.PropertyField(primitiveInitSize);
			if (targetMode.enumValueIndex != 0)
				EditorGUILayout.PropertyField(initPrimitivesOnMeshSurface);
			if (targetMode.enumValueIndex == 2)
			{
				if (initPrimitivesOnMeshSurface.boolValue == true)
				{
					EditorGUILayout.PropertyField(colmapOnePrimitivePerInitPoint);
					EditorGUILayout.PropertyField(colmapUseInitPointDistances);
					if (colmapUseInitPointDistances.boolValue == true)
						EditorGUILayout.PropertyField(colmapMaxInitPointDistance);
					EditorGUILayout.PropertyField(colmapInitPointsTrimAABBCenterSize);
				}
				else
					EditorGUILayout.PropertyField(colmapUseCameraBounds);
			}
			if (optimPrimitive.enumValueIndex != 5 && optimPrimitive.enumValueIndex != 6 && optimPrimitive.enumValueIndex != 3)
				EditorGUILayout.PropertyField(sphericalHarmonicsMode);
			EditorGUILayout.PropertyField(transparencyMode);
			if (transparencyMode.enumValueIndex != 0)
			{
				EditorGUILayout.PropertyField(maxFragmentsPerPixel);
				EditorGUILayout.PropertyField(alphaContributingCutoff);
			}
		}
		if (optimPrimitive.enumValueIndex == 3)
		{
			EditorGUILayout.PropertyField(voxelGridMaxResolution);
			EditorGUILayout.PropertyField(alphaContributingCutoff);
		}
		if (optimPrimitive.enumValueIndex == 6)
		{
			EditorGUILayout.PropertyField(meshInitState);
			EditorGUILayout.PropertyField(meshCatmullClarkSubdivisions);
			EditorGUILayout.PropertyField(meshCatmullClarkUseCreases);
			EditorGUILayout.PropertyField(meshInitSize);
		}
		if (optimPrimitive.enumValueIndex == 5 || optimPrimitive.enumValueIndex == 6)
		{
			EditorGUILayout.PropertyField(materialResolution);
			EditorGUILayout.PropertyField(materialHeightMode);
			if (optimPrimitive.enumValueIndex ==  5 && materialHeightMode.enumValueIndex == 2)
				EditorGUILayout.PropertyField(meshCatmullClarkSubdivisions);
		}
		if (targetMode.enumValueIndex == 1 || targetMode.enumValueIndex == 3)
			EditorGUILayout.PropertyField(randomViewZoomRange);

		// Controls
		EditorGUILayout.Space();
		EditorGUILayout.LabelField("Controls", EditorStyles.boldLabel);
		if (GUILayout.Button("Reset"))
			reset.boolValue = true;
		if (pause.boolValue == true)
			if (GUILayout.Button("Step Forward"))
				stepForward.boolValue = true;
		EditorGUILayout.PropertyField(pause);
		EditorGUILayout.PropertyField(displayMode);
		EditorGUILayout.PropertyField(separateFreeViewCamera);
		if (optimPrimitive.enumValueIndex == 6)
			EditorGUILayout.PropertyField(displayCatmullClarkControlCage);
		if ((optimPrimitive.enumValueIndex == 0 || optimPrimitive.enumValueIndex == 1 || optimPrimitive.enumValueIndex == 2 || optimPrimitive.enumValueIndex == 4) && GUILayout.Button("Double Primitive Count"))
		{
			needsToDoublePrimitives.boolValue = true;
			//globalLearningRate.floatValue *= 0.85f;
		}
		if ((optimPrimitive.enumValueIndex == 5 || optimPrimitive.enumValueIndex == 6) && GUILayout.Button("Double Texture Resolution"))
		{
			materialResolution.intValue *= 2;
			globalLearningRate.floatValue *= 0.85f;
			learningRatePosition.floatValue *= 0.7f;
		}
		if (optimPrimitive.enumValueIndex == 3 && GUILayout.Button("Double Volume Resolution"))
		{
			needsToDoubleVolumeRes.boolValue = true;
		}
		if (pause.boolValue == true && GUILayout.Button("Custom Export Button"))
		{
			needsToCustomExport.boolValue = true;
		}

		// Scheduling Controls
		EditorGUILayout.Space();
		EditorGUILayout.LabelField("Optimizer Scheduling", EditorStyles.boldLabel);
		EditorGUILayout.PropertyField(lrGlobalStartMulAndSpeed);
		if (optimPrimitive.enumValueIndex != 3)
			EditorGUILayout.PropertyField(lrGeometryStartMulAndSpeed);
		if (optimPrimitive.enumValueIndex == 6)
			EditorGUILayout.PropertyField(lrRegulStartMulAndSpeed);
		if (optimPrimitive.enumValueIndex == 0 || optimPrimitive.enumValueIndex == 1 || optimPrimitive.enumValueIndex == 2 || optimPrimitive.enumValueIndex == 3 || optimPrimitive.enumValueIndex == 4)
		{
			EditorGUILayout.PropertyField(doublingAmount);
			EditorGUILayout.PropertyField(doubleEveryXSteps);
		}

		// Optimizer settings
		EditorGUILayout.Space();
		EditorGUILayout.LabelField("Optimizer Settings", EditorStyles.boldLabel);
		EditorGUILayout.PropertyField(optimizer);
		EditorGUILayout.PropertyField(gradientsWarmupSteps);
		EditorGUILayout.PropertyField(optimSupersampling);
		EditorGUILayout.PropertyField(doAlphaLoss);
		if (targetMode.enumValueIndex == 2)
			EditorGUILayout.PropertyField(doAllInputFramesForEachOptimStep);
		if (targetMode.enumValueIndex != 0)
			EditorGUILayout.PropertyField(viewsPerOptimStep);
		EditorGUILayout.PropertyField(antitheticMutationsPerFrame);
		EditorGUILayout.PropertyField(parameterSeparationMode);

		// Optimizer controls
		EditorGUILayout.Space();
		EditorGUILayout.LabelField("Optimizer Controls", EditorStyles.boldLabel);
		EditorGUILayout.PropertyField(globalLearningRate);
		EditorGUILayout.PropertyField(beta1);
		EditorGUILayout.PropertyField(beta2);
		if (optimPrimitive.enumValueIndex != 3 && optimPrimitive.enumValueIndex != 5)
		{
			EditorGUILayout.PropertyField(learningRatePosition);
		}
		if (optimPrimitive.enumValueIndex == 4)
		{
			EditorGUILayout.PropertyField(learningRateRotation);
			EditorGUILayout.PropertyField(learningRateScale);
		}
		EditorGUILayout.PropertyField(learningRateColor);
		if ((transparencyMode.enumValueIndex != 0 || optimPrimitive.enumValueIndex == 3) && optimPrimitive.enumValueIndex != 6)
		{
			EditorGUILayout.PropertyField(learningRateAlpha);
		}
		if (optimPrimitive.enumValueIndex == 5 || optimPrimitive.enumValueIndex == 6)
		{
			EditorGUILayout.PropertyField(learningRateNormal);
			EditorGUILayout.PropertyField(learningRateMetallic);
			EditorGUILayout.PropertyField(learningRateRoughness);
			if (materialHeightMode.enumValueIndex != 0)
				EditorGUILayout.PropertyField(learningRateHeight);
		}
		if (optimPrimitive.enumValueIndex == 6)
		{
			//EditorGUILayout.PropertyField(learningRateUV);
			if (meshCatmullClarkUseCreases.boolValue == true)
				EditorGUILayout.PropertyField(learningRateCrease);
			EditorGUILayout.PropertyField(vertexRegularizerWeight);
			EditorGUILayout.PropertyField(vertexRegularizer2Weight);
		}

		// Resampling settings
		if (optimPrimitive.enumValueIndex == 0 || optimPrimitive.enumValueIndex == 1 || optimPrimitive.enumValueIndex == 2 || optimPrimitive.enumValueIndex == 4)
		{
			EditorGUILayout.Space();
			EditorGUILayout.LabelField("Resampling Settings", EditorStyles.boldLabel);
			EditorGUILayout.PropertyField(doPrimitiveResampling);
			EditorGUILayout.PropertyField(resamplingInterval);
			EditorGUILayout.PropertyField(optimStepsUnseenBeforeKill);
			EditorGUILayout.PropertyField(minPrimitiveWorldArea);
		}

		ValidateParameters();

		serializedObject.ApplyModifiedProperties();
	}
}
