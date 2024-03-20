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
	SerializedProperty colmapRescaler;
	SerializedProperty colmapUseMasking;
	SerializedProperty optimPrimitive;
	SerializedProperty primitiveCount;
	SerializedProperty primitiveInitSize;
	SerializedProperty primitiveInitSeed;
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
	SerializedProperty randomViewZoomRange;
	SerializedProperty backgroundMode;
	SerializedProperty backgroundColor;
	SerializedProperty envMapResolution;
	SerializedProperty displayAdaptiveTriangleBlurring;
	SerializedProperty optimAdaptiveTriangleBlurring;

	SerializedProperty reset;
	SerializedProperty pause;
	SerializedProperty stepForward;
	SerializedProperty displayMode;
	SerializedProperty separateFreeViewCamera;
	SerializedProperty debugTriangleView;

	SerializedProperty lrGlobalStartMulAndSpeed;
	SerializedProperty lrGeometryStartMulAndSpeed;
	SerializedProperty primitiveDoublingCountAndInterval;
	SerializedProperty resolutionDoublingCountAndInterval;

	SerializedProperty optimizer;
	SerializedProperty lossMode;
	SerializedProperty gradientsWarmupSteps;
	SerializedProperty optimResolutionFactor;
	SerializedProperty trianglePerVertexError;
	SerializedProperty pixelCountNormalization;
	SerializedProperty doAlphaLoss;
	SerializedProperty doStructuralLoss;
	SerializedProperty doStructuralWelding;
	SerializedProperty doAllInputFramesForEachOptimStep;
	SerializedProperty viewsPerOptimStep;
	SerializedProperty antitheticMutationsPerFrame;
	SerializedProperty parameterSeparationMode;

	SerializedProperty globalLearningRate;
	SerializedProperty beta1;
	SerializedProperty beta2;
	SerializedProperty learningRatePosition;
	SerializedProperty learningRateColor;
	SerializedProperty learningRateAlpha;
	SerializedProperty learningRateEnvMap;
	SerializedProperty structuralLossWeight;
	SerializedProperty structuralLossDistFactor;
	SerializedProperty structuralWeldDistFactor;

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
		colmapRescaler = serializedObject.FindProperty("colmapRescaler");
		colmapUseMasking = serializedObject.FindProperty("colmapUseMasking");
		optimPrimitive = serializedObject.FindProperty("optimPrimitive");
		primitiveCount = serializedObject.FindProperty("primitiveCount");
		primitiveInitSize = serializedObject.FindProperty("primitiveInitSize");
		primitiveInitSeed = serializedObject.FindProperty("primitiveInitSeed");
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
		randomViewZoomRange = serializedObject.FindProperty("randomViewZoomRange");
		backgroundMode = serializedObject.FindProperty("backgroundMode");
		backgroundColor = serializedObject.FindProperty("backgroundColor");
		envMapResolution = serializedObject.FindProperty("envMapResolution");
		displayAdaptiveTriangleBlurring = serializedObject.FindProperty("displayAdaptiveTriangleBlurring");
		optimAdaptiveTriangleBlurring = serializedObject.FindProperty("optimAdaptiveTriangleBlurring");

		reset = serializedObject.FindProperty("reset");
		pause = serializedObject.FindProperty("pause");
		stepForward = serializedObject.FindProperty("stepForward");
		displayMode = serializedObject.FindProperty("displayMode");
		separateFreeViewCamera = serializedObject.FindProperty("separateFreeViewCamera");
		debugTriangleView = serializedObject.FindProperty("debugTriangleView");

		lrGlobalStartMulAndSpeed = serializedObject.FindProperty("lrGlobalStartMulAndSpeed");
		lrGeometryStartMulAndSpeed = serializedObject.FindProperty("lrGeometryStartMulAndSpeed");
		primitiveDoublingCountAndInterval = serializedObject.FindProperty("primitiveDoublingCountAndInterval");
		resolutionDoublingCountAndInterval = serializedObject.FindProperty("resolutionDoublingCountAndInterval");

		optimizer = serializedObject.FindProperty("optimizer");
		lossMode = serializedObject.FindProperty("lossMode");
		gradientsWarmupSteps = serializedObject.FindProperty("gradientsWarmupSteps");
		optimResolutionFactor = serializedObject.FindProperty("optimResolutionFactor");
		trianglePerVertexError = serializedObject.FindProperty("trianglePerVertexError");
		pixelCountNormalization = serializedObject.FindProperty("pixelCountNormalization");
		doAlphaLoss = serializedObject.FindProperty("doAlphaLoss");
		doStructuralLoss = serializedObject.FindProperty("doStructuralLoss");
		doStructuralWelding = serializedObject.FindProperty("doStructuralWelding");
		doAllInputFramesForEachOptimStep = serializedObject.FindProperty("doAllInputFramesForEachOptimStep");
		viewsPerOptimStep = serializedObject.FindProperty("viewsPerOptimStep");
		antitheticMutationsPerFrame = serializedObject.FindProperty("antitheticMutationsPerFrame");
		parameterSeparationMode = serializedObject.FindProperty("parameterSeparationMode");

		globalLearningRate = serializedObject.FindProperty("globalLearningRate");
		beta1 = serializedObject.FindProperty("beta1");
		beta2 = serializedObject.FindProperty("beta2");
		learningRatePosition = serializedObject.FindProperty("learningRatePosition");
		learningRateColor = serializedObject.FindProperty("learningRateColor");
		learningRateAlpha = serializedObject.FindProperty("learningRateAlpha");
		learningRateEnvMap = serializedObject.FindProperty("learningRateEnvMap");
		structuralLossWeight = serializedObject.FindProperty("structuralLossWeight");
		structuralLossDistFactor = serializedObject.FindProperty("structuralLossDistFactor");
		structuralWeldDistFactor = serializedObject.FindProperty("structuralWeldDistFactor");

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
		optimResolutionFactor.floatValue = math.max(optimResolutionFactor.floatValue, 1.0f / 8.0f);
		viewsPerOptimStep.intValue = math.max(viewsPerOptimStep.intValue, 1);
		targetResolution.vector2IntValue = new Vector2Int(math.max(targetResolution.vector2IntValue.x, 2), math.max(targetResolution.vector2IntValue.y, 2));
		maxFragmentsPerPixel.intValue = math.max(maxFragmentsPerPixel.intValue, 1);
		maxFragmentsPerPixel.intValue = math.min(maxFragmentsPerPixel.intValue, 64);
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
		EditorGUILayout.PropertyField(optimPrimitive);
		EditorGUILayout.PropertyField(primitiveInitSeed);
		if (optimPrimitive.enumValueIndex == 0 || optimPrimitive.enumValueIndex == 1 || optimPrimitive.enumValueIndex == 2)
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
		if (targetMode.enumValueIndex == 1 || targetMode.enumValueIndex == 3)
			EditorGUILayout.PropertyField(randomViewZoomRange);
		EditorGUILayout.PropertyField(backgroundMode);
		if (backgroundMode.enumValueIndex == 0)
			EditorGUILayout.PropertyField(backgroundColor);
		if (backgroundMode.enumValueIndex == 2)
			EditorGUILayout.PropertyField(envMapResolution);
		EditorGUILayout.PropertyField(displayAdaptiveTriangleBlurring);
		EditorGUILayout.PropertyField(optimAdaptiveTriangleBlurring);

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
		EditorGUILayout.PropertyField(debugTriangleView);
		if ((optimPrimitive.enumValueIndex == 0 || optimPrimitive.enumValueIndex == 1 || optimPrimitive.enumValueIndex == 2 || optimPrimitive.enumValueIndex == 4) && GUILayout.Button("Double Primitive Count"))
			needsToDoublePrimitives.boolValue = true;
		if (optimPrimitive.enumValueIndex == 3 && GUILayout.Button("Double Volume Resolution"))
			needsToDoubleVolumeRes.boolValue = true;
		if (pause.boolValue == true && GUILayout.Button("Custom Export Button"))
			needsToCustomExport.boolValue = true;

		// Scheduling Controls
		EditorGUILayout.Space();
		EditorGUILayout.LabelField("Optimizer Scheduling", EditorStyles.boldLabel);
		EditorGUILayout.PropertyField(lrGlobalStartMulAndSpeed);
		if (optimPrimitive.enumValueIndex != 3)
			EditorGUILayout.PropertyField(lrGeometryStartMulAndSpeed);
		EditorGUILayout.PropertyField(primitiveDoublingCountAndInterval);
		EditorGUILayout.PropertyField(resolutionDoublingCountAndInterval);

		// Optimizer settings
		EditorGUILayout.Space();
		EditorGUILayout.LabelField("Optimizer Settings", EditorStyles.boldLabel);
		EditorGUILayout.PropertyField(optimizer);
		EditorGUILayout.PropertyField(lossMode);
		EditorGUILayout.PropertyField(gradientsWarmupSteps);
		EditorGUILayout.PropertyField(optimResolutionFactor);
		EditorGUILayout.PropertyField(trianglePerVertexError);
		EditorGUILayout.PropertyField(pixelCountNormalization);
		EditorGUILayout.PropertyField(doAlphaLoss);
		EditorGUILayout.PropertyField(doStructuralLoss);
		EditorGUILayout.PropertyField(doStructuralWelding);
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
		EditorGUILayout.PropertyField(learningRatePosition);
		EditorGUILayout.PropertyField(learningRateColor);
		if ((transparencyMode.enumValueIndex != 0 || optimPrimitive.enumValueIndex == 3) && optimPrimitive.enumValueIndex != 6)
			EditorGUILayout.PropertyField(learningRateAlpha);
		if (backgroundMode.enumValueIndex == 2)
			EditorGUILayout.PropertyField(learningRateEnvMap);
		if (doStructuralLoss.boolValue == true)
		{
			EditorGUILayout.PropertyField(structuralLossWeight);
			EditorGUILayout.PropertyField(structuralLossDistFactor);
		}
		if (doStructuralWelding.boolValue == true)
		{
			EditorGUILayout.PropertyField(structuralWeldDistFactor);
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
