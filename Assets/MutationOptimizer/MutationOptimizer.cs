using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
#if UNITY_EDITOR
using Unity.Burst;
#endif
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;


//[ExecuteInEditMode]
public class MutationOptimizer : MonoBehaviour
{
	// ======================= PRIVATE VARIABLES =======================
	public RenderTexture optimRenderTargetMutatedMinus;
	public RenderTexture optimRenderTarget;
	public RenderTexture resolvedFrameFreeView;
	public RenderTexture resolvedFrameMutatedMinus;
	public RenderTexture resolvedFrameMutatedPlus;
	public RenderTexture tempResolvedFrameBuffer;
	public RenderTexture targetFrameBuffer;
	private Material rasterMaterial;
	private Material envMapMaterial;
	private Material blurringMaterial;
	private Material adaptiveTriangleBlurMaterial;
	private Camera cameraDisplay;
	private Camera cameraOptim;
	private ComputeShader primitiveRendererCS;
	private ComputeShader mutationOptimizerCS;
	private ComputeBuffer[] primitiveBuffer;
	private ComputeBuffer[] primitiveBufferMutated;
	private ComputeBuffer[] optimStepGradientsBuffer;
	private ComputeBuffer[] optimStepMutationError;
	private ComputeBuffer[] gradientMoments1Buffer;
	private ComputeBuffer[] gradientMoments2Buffer;
	private ComputeBuffer[] optimStepCounterBuffer;
	private ComputeBuffer primitiveKillCounters;
	private ComputeBuffer appendValidIDsBuffer;
	private ComputeBuffer appendInvalidIDsBuffer;
	private ComputeBuffer argsValidIDsBuffer;
	private ComputeBuffer argsInvalidIDsBuffer;
	private ComputeBuffer argsResampling;
	private ComputeBuffer perPixelFragmentCounterBuffer;
	private ComputeBuffer perPixelFragmentListBuffer;
	private GraphicsBuffer sortedPrimitiveIDBuffer;
	private GraphicsBuffer sortedPrimitiveDistanceBuffer;
	private ComputeBuffer sortedValidPrimitiveIDBuffer;
	private ComputeBuffer structuralEdgeClosestNeighbourBuffer;
	private ComputeBuffer structuralVertexWeldingBuffer;
	private Bounds mesh3DSceneBounds;
	private Texture2D[] colmapViewsTarget;
	private Vector3[] colmapViewsPos;
	private Quaternion[] colmapViewsRot;
	private float3[] colmapInitPointPositions;
	private float[] colmapInitPointMinDistances;
	private Bounds colmapInitPointBounds;
	private Vector2Int colmapViewResolution;
	private int currentViewPoint = 0;
	public int currentOptimStep = 0;
	private int currentStochasticFrame = 0;
	private Color depthIDClearColor = new Color(4294967295, 0, 0, 0);
	private Stopwatch systemTimer = new Stopwatch();
	private int optimStepsSeparateCount = 1;
	private float2 learningRateGlobalStartEnd = -1.0f;
	private float2 learningRatePositionStartEnd = -1.0f;
	private int primitiveDoublingCounter = 0;
	private int resolutionDoublingCounter = 0;
	private Bounds colmapTrimBounds;
	private string exportPath = "";
	private bool isDoingWarmup = false;
	public GpuSorting m_Sorter;
	public GpuSorting.Args m_SorterArgs;

	// Compute kernels
	private int kernelComputeL2Image;
	private int kernelClearRenderTarget;
	private int kernelResolveOpaqueRender;
	private int kernelResetAlphaCounters;
	private int kernelSortAlphaFragments;
	private int kernelInitBitonicSortPrimitives;
	private int kernelBitonicSortPrimitives;
	private int kernelFindContributingAlphaFragments;
	private int kernelResetMutationLossAccumulation;
	private int kernelAccumulateMutationLoss;
	private int kernelAccumulateMutationGradientsResetLoss;
	private int kernelApplyRandomMutation;
	private int kernelCreateNewRandomMutation;
	private int kernelMaintainClosestEdgeNeighbours;
	private int kernelUpdateClosestEdgeNeighbours;
	private int kernelAccumulateMutationLossStructural;
	private int kernelWeldVertices;
	private int kernelEnvMapResetMutationLossAccumulation;
	private int kernelEnvMapAccumulateMutationLoss;
	private int kernelEnvMapAccumulateMutationGradientsResetLoss;
	private int kernelEnvMapApplyRandomMutation;
	private int kernelEnvMapCreateNewRandomMutation;
	private int kernelResetVisibilityCounter;
	private int kernelDecrementVisibilityCounter;
	private int kernelListValidAndInvalidPrimitiveIDs;
	private int kernelInitArgsResampling;
	private int kernelInitBitonicSortValidPrimitives;
	private int kernelBitonicSortValidPrimitives;
	private int kernelPairResampling;
	private int kernelTripleResampling;

	// Variable change watchers
	private int setPrimitiveCount = -1;
	private int setResolutionX = -1;
	private int setResolutionY = -1;
	private TargetMode setTargetMode = TargetMode.Model;
	private PrimitiveType setOptimPrimitive = PrimitiveType.TrianglesSolidUnlit;
	private bool setPrimitiveResampling = false;
	private float setOptimSupersampling = -1;
	private TransparencyMode setTransparencyMode = TransparencyMode.None;
	private int setMaxFragmentsPerPixel = -1;
	[HideInInspector] public bool dummy = true;
	public bool needsToDoublePrimitives = false;

	// ======================= INTERFACE =======================
	public Vector2Int targetResolution = new Vector2Int(512, 512);
	public TargetMode targetMode = TargetMode.Model;
	public Texture2D target2DImage;
	public GameObject target3DMesh;
	public string target3DCOLMAPFolder;
	public float colmapRescaler = 1.0f;
	public bool colmapUseMasking = false;
	public PrimitiveType optimPrimitive = PrimitiveType.TrianglesSolidUnlit;
	public int primitiveCount = 1;
	public float primitiveInitSize = 1.0f;
	public int primitiveInitSeed = -1;
	public bool initPrimitivesOnMeshSurface = false;
	public bool colmapUseCameraBounds = false;
	public bool colmapOnePrimitivePerInitPoint = false;
	public bool colmapUseInitPointDistances = false;
	public float colmapMaxInitPointDistance = 0.5f;
	public Vector4 colmapInitPointsTrimAABBCenterSize = Vector4.zero;
	public SphericalHarmonicsMode sphericalHarmonicsMode = SphericalHarmonicsMode.None;
	public TransparencyMode transparencyMode = TransparencyMode.None;
	public int maxFragmentsPerPixel = 1;
	[Range(0.0f, 1.0f)] public float alphaContributingCutoff = 0.9f;
	public Vector2 randomViewZoomRange = Vector2.one;
	public BackgroundMode backgroundMode = BackgroundMode.Color;
	public Color backgroundColor = Color.black;
	public int envMapResolution = 256;
	public bool displayAdaptiveTriangleBlurring = false;
	public bool optimAdaptiveTriangleBlurring = false;

	public bool reset = false;
	public bool pause = false;
	public bool stepForward = false;
	public DisplayMode displayMode = DisplayMode.Optimization;
	public bool separateFreeViewCamera = true;
	public bool debugTriangleView = false;

	public Vector2 lrGlobalStartMulAndSpeed = Vector2.zero;
	public Vector2 lrGeometryStartMulAndSpeed = Vector2.zero;
	public Vector2Int primitiveDoublingCountAndInterval = Vector2Int.zero;
	public Vector2Int resolutionDoublingCountAndInterval = Vector2Int.zero;

	public Optimizer optimizer = Optimizer.Adam;
	public LossMode lossMode = LossMode.L2;
	public int gradientsWarmupSteps = 16;
	public float optimResolutionFactor = 1;
	public bool trianglePerVertexError = false;
	public bool pixelCountNormalization = false;
	public bool doAlphaLoss = true;
	public bool doStructuralLoss = true;
	public bool doStructuralWelding = true;
	public bool doAllInputFramesForEachOptimStep = false;
	public int viewsPerOptimStep = 1;
	public int antitheticMutationsPerFrame = 16;
	public ParameterOptimSeparationMode parameterSeparationMode = ParameterOptimSeparationMode.None;
	public int debugManualColmapChoice = 35;

	[Range(0.0f, 2.0f)] public float globalLearningRate = 1.0f;
	[Range(0.0f, 1.0f)] public float beta1 = 0.9f;
	[Range(0.0f, 1.0f)] public float beta2 = 0.999f;
	[LogarithmicRange(0.0f, 0.001f, 1.0f)] public float learningRatePosition = 0.01f;
	[LogarithmicRange(0.0f, 0.001f, 1.0f)] public float learningRateColor = 0.01f;
	[LogarithmicRange(0.0f, 0.001f, 1.0f)] public float learningRateAlpha = 0.01f;
	[LogarithmicRange(0.0f, 0.001f, 1.0f)] public float learningRateEnvMap = 0.01f;
	[Range(0.0f, 1.0f)] public float structuralLossWeight = 0.5f;
	[Range(0.0f, 2.0f)] public float structuralLossDistFactor = 1.0f;
	[Range(0.0f, 2.0f)] public float structuralWeldDistFactor = 1.0f;

	public bool doPrimitiveResampling = true;
	public int resamplingInterval = 1;
	public int optimStepsUnseenBeforeKill = 16;
	public float minPrimitiveWorldArea = 0.0001f;

	public float millisecondsPerOptimStep = 0.0f;
	public float totalElapsedSeconds = 0.0f;
	public Vector2Int internalOptimResolution = Vector2Int.one;
	public Vector3Int voxelGridResolution = Vector3Int.one;
	public int colmapImageCount = 0;




	// =========================== UNITY ===========================
	void OnEnable()
	{
		System.Globalization.CultureInfo.DefaultThreadCurrentCulture = System.Globalization.CultureInfo.InvariantCulture;
		System.Globalization.CultureInfo.DefaultThreadCurrentUICulture = System.Globalization.CultureInfo.InvariantCulture;
		Physics.simulationMode = SimulationMode.Script;
		primitiveRendererCS = (ComputeShader)Resources.Load("PrimitiveRendererTools");
		mutationOptimizerCS = (ComputeShader)Resources.Load("MutationOptimizer");
		cameraDisplay = GameObject.Find("CameraDisplay").GetComponent<Camera>();
		cameraOptim = GameObject.Find("CameraOptim").GetComponent<Camera>();
		blurringMaterial = new Material(Shader.Find("Custom/Blurring"));
		blurringMaterial.hideFlags = HideFlags.HideAndDontSave;
		adaptiveTriangleBlurMaterial = new Material(Shader.Find("Custom/AdaptiveTriangleBlur"));
		adaptiveTriangleBlurMaterial.hideFlags = HideFlags.HideAndDontSave;

		kernelClearRenderTarget = primitiveRendererCS.FindKernel("ClearRenderTarget");
		kernelResolveOpaqueRender = primitiveRendererCS.FindKernel("ResolveOpaqueRender");
		kernelComputeL2Image = primitiveRendererCS.FindKernel("ComputeL2Image");
		kernelResetAlphaCounters = primitiveRendererCS.FindKernel("ResetAlphaCounters");
		kernelSortAlphaFragments = primitiveRendererCS.FindKernel("SortAlphaFragments");
		kernelInitBitonicSortPrimitives = primitiveRendererCS.FindKernel("InitBitonicSortPrimitives");
		kernelBitonicSortPrimitives = primitiveRendererCS.FindKernel("BitonicSortPrimitives");
		kernelFindContributingAlphaFragments = primitiveRendererCS.FindKernel("FindContributingAlphaFragments");
		kernelResetMutationLossAccumulation = mutationOptimizerCS.FindKernel("ResetMutationLossAccumulation");
		kernelAccumulateMutationLoss = mutationOptimizerCS.FindKernel("AccumulateMutationLoss");
		kernelAccumulateMutationGradientsResetLoss = mutationOptimizerCS.FindKernel("AccumulateMutationGradientsResetLoss");
		kernelApplyRandomMutation = mutationOptimizerCS.FindKernel("ApplyRandomMutation");
		kernelCreateNewRandomMutation = mutationOptimizerCS.FindKernel("CreateNewRandomMutation");
		//kernelMaintainClosestEdgeNeighbours = mutationOptimizerCS.FindKernel("MaintainClosestEdgeNeighbours");
		//kernelUpdateClosestEdgeNeighbours = mutationOptimizerCS.FindKernel("UpdateClosestEdgeNeighbours");
		kernelAccumulateMutationLossStructural = mutationOptimizerCS.FindKernel("AccumulateMutationLossStructural");
		kernelWeldVertices = mutationOptimizerCS.FindKernel("WeldVertices");
		kernelEnvMapAccumulateMutationLoss = mutationOptimizerCS.FindKernel("EnvMapAccumulateMutationLoss");
		kernelEnvMapAccumulateMutationGradientsResetLoss = mutationOptimizerCS.FindKernel("EnvMapAccumulateMutationGradientsResetLoss");
		kernelEnvMapApplyRandomMutation = mutationOptimizerCS.FindKernel("EnvMapApplyRandomMutation");
		kernelEnvMapCreateNewRandomMutation = mutationOptimizerCS.FindKernel("EnvMapCreateNewRandomMutation");
		kernelResetVisibilityCounter = mutationOptimizerCS.FindKernel("ResetVisibilityCounter");
		kernelDecrementVisibilityCounter = mutationOptimizerCS.FindKernel("DecrementVisibilityCounter");
		kernelListValidAndInvalidPrimitiveIDs = mutationOptimizerCS.FindKernel("ListValidAndInvalidPrimitiveIDs");
		kernelInitArgsResampling = mutationOptimizerCS.FindKernel("InitArgsResampling");
		kernelInitBitonicSortValidPrimitives = mutationOptimizerCS.FindKernel("InitBitonicSortValidPrimitives");
		kernelBitonicSortValidPrimitives = mutationOptimizerCS.FindKernel("BitonicSortValidPrimitives");
		kernelPairResampling = mutationOptimizerCS.FindKernel("PairResampling");
		kernelTripleResampling = mutationOptimizerCS.FindKernel("TripleResampling");

		int primitiveGroupCount = backgroundMode == BackgroundMode.EnvMap ? 2 : 1;
		primitiveBuffer = new ComputeBuffer[primitiveGroupCount];
		optimStepGradientsBuffer = new ComputeBuffer[primitiveGroupCount];
		gradientMoments1Buffer = new ComputeBuffer[primitiveGroupCount];
		gradientMoments2Buffer = new ComputeBuffer[primitiveGroupCount];
		primitiveBufferMutated = new ComputeBuffer[primitiveGroupCount];
		optimStepMutationError = new ComputeBuffer[primitiveGroupCount];
		optimStepCounterBuffer = new ComputeBuffer[primitiveGroupCount];

		needsToDoublePrimitives = false;

		if (learningRatePositionStartEnd.x < 0.0f)
		{
			learningRatePositionStartEnd = new float2(learningRatePosition * lrGeometryStartMulAndSpeed.x, learningRatePosition);
		}
		if (learningRateGlobalStartEnd.x < 0.0f)
		{
			learningRateGlobalStartEnd = new float2(globalLearningRate * lrGlobalStartMulAndSpeed.x, globalLearningRate);
		}

		colmapTrimBounds = new Bounds(new Vector3(colmapInitPointsTrimAABBCenterSize.x, colmapInitPointsTrimAABBCenterSize.y, colmapInitPointsTrimAABBCenterSize.z), Vector3.one * colmapInitPointsTrimAABBCenterSize.w);
	}

	void Update()
	{
		// Keyboard controls
		if (Input.GetKeyDown(KeyCode.R)) reset = true;
		if (Input.GetKeyDown(KeyCode.P)) pause = !pause;
		if (Input.GetKeyDown(KeyCode.Space)) stepForward = true;
		if (Input.GetKeyDown(KeyCode.F)) separateFreeViewCamera = !separateFreeViewCamera;
		if (Input.GetKeyDown(KeyCode.F1)) displayMode = DisplayMode.Optimization;
		if (Input.GetKeyDown(KeyCode.F2)) displayMode = DisplayMode.Target;
		if (Input.GetKeyDown(KeyCode.T)) debugTriangleView = !debugTriangleView;

		// Handle critical parameter change (reset everything and restart)
		if (ChecksResetEverythingConditions() == true || reset == true)
		{
			reset = false;
			ResetWatchVariables();
			ResetEverything();
			SetOptimizerLearningRatesAndGetSeparateCount(0); // For getting the separate parameter group count early
			totalElapsedSeconds = 0.0f;
			systemTimer.Restart();
			isDoingWarmup = true;
		}

		// Handle live optim param change that affects internal resolution (reset optim state and buffers)
		if (setOptimSupersampling != optimResolutionFactor || setTransparencyMode != transparencyMode || setMaxFragmentsPerPixel != maxFragmentsPerPixel || setResolutionX != targetResolution.x || setResolutionY != targetResolution.y)
		{
			InitAllOptimBuffers();
			ResetKeywords(primitiveRendererCS, true, true, true);
			ResetKeywords(mutationOptimizerCS, true, true, true);
			ResetKeywords(rasterMaterial, true, true, true);
			ResetKeywords(adaptiveTriangleBlurMaterial, true, true, true);
			ResetOptimizationStep(0);
			if (backgroundMode == BackgroundMode.EnvMap)
				ResetOptimizationStep(1);
		}

		// Handle live re-compile of compute shader
		if (CheckKeywordsValidity(primitiveRendererCS, true, true, true) == false || CheckKeywordsValidity(mutationOptimizerCS, true, true, true) == false)
		{
			ResetKeywords(primitiveRendererCS, true, true, true);
			ResetKeywords(mutationOptimizerCS, true, true, true);
			ResetKeywords(rasterMaterial, true, true, true);
			ResetKeywords(adaptiveTriangleBlurMaterial, true, true, true);
		}

		// TEMP HACK WARMUP
		if (isDoingWarmup == true && currentOptimStep >= gradientsWarmupSteps)
		{
			currentOptimStep = 0;
			currentViewPoint = 0;
			isDoingWarmup = false;
			totalElapsedSeconds = 0.0f;
		}
		mutationOptimizerCS.SetFloat("_DoGradientDescent", isDoingWarmup == true ? 0.0f : 1.0f);

		if (currentOptimStep == 0)
			systemTimer.Restart();

		// Scheduling update
		UpdateSchedulingStuff();

		// Optimization Loop
		OptimizationUpdate();

		// Update display
		DisplayModeUpdate();
	}

	public void OptimizationUpdate()
	{
		if (pause == true && stepForward == false)
		{
			systemTimer.Restart();
			return;
		}

		// Prepare parameters
		SetSharedComputeFrameParameters(primitiveRendererCS);
		SetSharedComputeFrameParameters(mutationOptimizerCS);

		// Decrement visibility counters every optim step
		if (doPrimitiveResampling == true)
		{
			mutationOptimizerCS.SetInt("_PrimitiveCount", primitiveBuffer[0].count);
			mutationOptimizerCS.SetBuffer(kernelDecrementVisibilityCounter, "_PrimitiveKillCounters", primitiveKillCounters);
			DispatchCompute1D(mutationOptimizerCS, kernelDecrementVisibilityCounter, primitiveCount, 256);
		}

		// Accumulate gradients for this optim step
		for (int i = 0; i < viewsPerOptimStep; i++)
		{
			// Set up new view point for optim 3D
			if (targetMode == TargetMode.Model)
			{
				RandomizeCameraView();
				cameraOptim.Render();
			}
			// Set up new view point for optim COLMAP
			else if (targetMode == TargetMode.COLMAP)
			{
				int randTarget = (int)(UnityEngine.Random.value * (colmapImageCount - 1));
				if (doAllInputFramesForEachOptimStep == true)
					randTarget = currentViewPoint % colmapImageCount;
				if (debugManualColmapChoice >= 0)
					randTarget = debugManualColmapChoice;
				Graphics.Blit(colmapViewsTarget[randTarget], targetFrameBuffer);
				cameraOptim.transform.position = colmapViewsPos[randTarget];
				cameraOptim.transform.rotation = colmapViewsRot[randTarget];
			}
			// Set up simple 2D image view
			else
			{
				cameraOptim.transform.position = new Vector3(0.5f, 0.5f, -2.0f);
				cameraOptim.transform.LookAt(new Vector3(0.5f, 0.5f, 0.5f));
				cameraOptim.orthographicSize = 0.5f;
			}
			SetCameraOptimMatrices();

			// Accumulate gradients for this view step
			mutationOptimizerCS.SetInt("_CurrentFrame", currentViewPoint);
			primitiveRendererCS.SetInt("_CurrentFrame", currentViewPoint);
			if (backgroundMode == BackgroundMode.RandomColor)
				primitiveRendererCS.SetVector("_BackgroundColor", UnityEngine.Random.ColorHSV());
			for (int j = 0; j < antitheticMutationsPerFrame; j++)
			{
				mutationOptimizerCS.SetInt("_CurrentMutation", j);
				rasterMaterial.SetInt("_CurrentMutation", j);

				// Accumulate gradients for this parameter grouping
				for (int k = 0; k < optimStepsSeparateCount; k++)
				{
					// Set separate optim learning rates
					SetOptimizerLearningRatesAndGetSeparateCount(k, false);

					// Minus Epsilon
					mutationOptimizerCS.SetFloat("_IsAntitheticMutation", 1.0f);
					CreateNewRandomMutation(0);
					if (backgroundMode == BackgroundMode.EnvMap && learningRateEnvMap > 0.0f)
						CreateNewRandomMutation(1);
					RenderProceduralPrimitivesOptimScene(cameraOptim, primitiveBufferMutated, resolvedFrameMutatedMinus, optimRenderTargetMutatedMinus, j == 0 ? true : false);

					// Plus Epsilon
					mutationOptimizerCS.SetFloat("_IsAntitheticMutation", 0.0f);
					CreateNewRandomMutation(0);
					if (backgroundMode == BackgroundMode.EnvMap && learningRateEnvMap > 0.0f)
						CreateNewRandomMutation(1);
					RenderProceduralPrimitivesOptimScene(cameraOptim, primitiveBufferMutated, resolvedFrameMutatedPlus, optimRenderTarget, false);

					// Accumulate Gradients
					AccumulateOptimizationStep(0, k);
					if (backgroundMode == BackgroundMode.EnvMap && learningRateEnvMap > 0.0f)
						AccumulateOptimizationStep(1, k);
				}
			}

			// Next camera position
			currentViewPoint += 1;
		}

		if (isDoingWarmup == false)
		{
			// Apply accumulated gradient for this optim step
			SetOptimizerLearningRatesAndGetSeparateCount(0, true); // Set full optim learning rates
			ResetKeywords(mutationOptimizerCS, true, true, true);
			ApplyOptimizationStep(0);
			if (currentOptimStep > 1)
				PerformPrimitiveResampling(0);
			ResetOptimizationStep(0);

			if (backgroundMode == BackgroundMode.EnvMap)
			{
				ApplyOptimizationStep(1);
				ResetOptimizationStep(1);
			}
		}

		// Metrics
		currentOptimStep += 1;
		millisecondsPerOptimStep = (float)systemTimer.Elapsed.TotalMilliseconds;
		totalElapsedSeconds += (float)systemTimer.Elapsed.TotalMilliseconds / 1000.0f;
		systemTimer.Restart();

		if (stepForward == true)
			stepForward = false;
	}

	public void DisplayModeUpdate()
	{
		// Update if changed
		if (targetMode == TargetMode.Image || separateFreeViewCamera == false)
		{
			cameraDisplay.cullingMask = 0;
			if (displayMode == DisplayMode.Optimization)
			{
				// Need to render the unmutated primitives from the optim camera here to display unperturbed, stable result for 2D image
				if (targetMode == TargetMode.Image)
				{
					RenderProceduralPrimitivesOptimScene(cameraOptim, primitiveBuffer, resolvedFrameFreeView, optimRenderTarget, false);
					cameraDisplay.GetComponent<DisplayRenderTexture>().displayRenderTexture = resolvedFrameFreeView;
				}
				else
				{
					cameraDisplay.GetComponent<DisplayRenderTexture>().displayRenderTexture = resolvedFrameMutatedPlus;
				}
			}
			else if (displayMode == DisplayMode.Target)
			{
				cameraDisplay.GetComponent<DisplayRenderTexture>().displayRenderTexture = targetFrameBuffer;
			}
		}
		else
		{
			if (displayMode == DisplayMode.Optimization)
			{
				cameraDisplay.GetComponent<DisplayRenderTexture>().displayRenderTexture = null;
				cameraDisplay.cullingMask = 0;
			}
			else if (displayMode == DisplayMode.Target)
			{
				cameraDisplay.GetComponent<DisplayRenderTexture>().displayRenderTexture = null;
				cameraDisplay.cullingMask = ~0;
			}
		}
	}

	void OnDisable()
	{
		ReleaseEverything();
	}

	void ReleaseEverything()
	{
		if (rasterMaterial != null)
		{
			if (Application.isPlaying)
			{
				Destroy(rasterMaterial);
			}
			else
			{
				DestroyImmediate(rasterMaterial);
			}
		}

		for (int i = 0; i < primitiveBuffer.Length; i++)
		{
			if (primitiveBuffer[i] != null)
				primitiveBuffer[i].Release();
			primitiveBuffer[i] = null;
		}
		ReleaseOptimBuffers();
		colmapViewsTarget = null;
	}

	void ReleaseOptimBuffers()
	{
		for (int i = 0; i < primitiveBufferMutated.Length; i++)
			if (primitiveBufferMutated[i] != null)
				primitiveBufferMutated[i].Release();
		if (optimRenderTarget != null)
			optimRenderTarget.Release();
		if (optimRenderTargetMutatedMinus != null)
			optimRenderTargetMutatedMinus.Release();
		if (resolvedFrameFreeView != null)
			resolvedFrameFreeView.Release();
		if (resolvedFrameMutatedMinus != null)
			resolvedFrameMutatedMinus.Release();
		if (resolvedFrameMutatedPlus != null)
			resolvedFrameMutatedPlus.Release();
		for (int i = 0; i < optimStepGradientsBuffer.Length; i++)
			if (optimStepGradientsBuffer[i] != null)
				optimStepGradientsBuffer[i].Release();
		for (int i = 0; i < optimStepMutationError.Length; i++)
			if (optimStepMutationError[i] != null)
				optimStepMutationError[i].Release();
		for (int i = 0; i < gradientMoments1Buffer.Length; i++)
			if (gradientMoments1Buffer[i] != null)
				gradientMoments1Buffer[i].Release();
		for (int i = 0; i < gradientMoments2Buffer.Length; i++)
			if (gradientMoments2Buffer[i] != null)
				gradientMoments2Buffer[i].Release();
		for (int i = 0; i < optimStepCounterBuffer.Length; i++)
			if (optimStepCounterBuffer[i] != null)
				optimStepCounterBuffer[i].Release();
		if (primitiveKillCounters != null)
			primitiveKillCounters.Release();
		if (appendValidIDsBuffer != null)
			appendValidIDsBuffer.Release();
		if (appendInvalidIDsBuffer != null)
			appendInvalidIDsBuffer.Release();
		if (argsValidIDsBuffer != null)
			argsValidIDsBuffer.Release();
		if (argsInvalidIDsBuffer != null)
			argsInvalidIDsBuffer.Release();
		if (perPixelFragmentCounterBuffer != null)
			perPixelFragmentCounterBuffer.Release();
		if (perPixelFragmentListBuffer != null)
			perPixelFragmentListBuffer.Release();
		if (sortedPrimitiveIDBuffer != null)
			sortedPrimitiveIDBuffer.Release();
		if (sortedPrimitiveDistanceBuffer != null)
			sortedPrimitiveDistanceBuffer.Release();
		if (sortedValidPrimitiveIDBuffer != null)
			sortedValidPrimitiveIDBuffer.Release();
		if (structuralEdgeClosestNeighbourBuffer != null)
			structuralEdgeClosestNeighbourBuffer.Release();
		if (structuralVertexWeldingBuffer != null)
			structuralVertexWeldingBuffer.Release();
		m_SorterArgs.resources.Dispose();
	}




	// ========================= RENDERING =========================
	void OnRenderObject()
	{
		// Only to render primitives in scene view
		if (rasterMaterial == null || primitiveBuffer[0] == null || Camera.current == cameraOptim || displayMode != DisplayMode.Optimization || separateFreeViewCamera == false || (transparencyMode == TransparencyMode.SortedAlpha && perPixelFragmentListBuffer == null))
			return;

		if (transparencyMode == TransparencyMode.None && displayAdaptiveTriangleBlurring == false)
		{
			Vector3 cameraPos = Camera.current.transform.position;
			Matrix4x4 cameraVP = GL.GetGPUProjectionMatrix(Camera.current.projectionMatrix, true) * Camera.current.worldToCameraMatrix;

			// Draw env map first
			if (backgroundMode == BackgroundMode.EnvMap)
			{
				Matrix4x4 invCameraVP = cameraVP.inverse;
				envMapMaterial.SetInt("_EnvMapResolution", envMapResolution);
				envMapMaterial.SetMatrix("_CameraInvVP", invCameraVP);
				envMapMaterial.SetVector("_CurrentCameraWorldPos", cameraPos);
				envMapMaterial.SetBuffer("_EnvMapPrimitiveBuffer", primitiveBuffer[1]);
				envMapMaterial.SetFloat("_DoColorOrViewDir", 0.0f);
				envMapMaterial.SetPass(0);
				GL.Begin(GL.QUADS);
				GL.Color(Color.red);
				GL.Vertex3(0, 0, 0);
				GL.Vertex3(0, 1, 0);
				GL.Vertex3(1, 1, 0);
				GL.Vertex3(1, 0, 0);
				GL.End();
			}

			rasterMaterial.SetFloat("_DebugTriangleView", debugTriangleView == true ? 1.0f : 0.0f);
			rasterMaterial.SetVector("_CurrentCameraWorldPos", cameraPos);
			rasterMaterial.SetFloat("_SimpleColorRender", 1.0f);
			rasterMaterial.SetInt("_PrimitiveCount", primitiveBuffer[0].count);
			rasterMaterial.SetMatrix("_CameraMatrixVP", cameraVP);
			rasterMaterial.SetBuffer("_PrimitiveBuffer", primitiveBuffer[0]);
			rasterMaterial.SetPass(0);
			Graphics.DrawProceduralNow(MeshTopology.Triangles, 3, primitiveCount);
		}
		else
		{
			RenderTexture cameraTarget = RenderTexture.active;
			primitiveRendererCS.SetVector("_RandomBackgroundColor", Color.black);
			RenderProceduralPrimitivesOptimScene(Camera.current, primitiveBuffer, resolvedFrameFreeView, optimRenderTarget);
			RenderTexture.active = cameraTarget;
			if (displayAdaptiveTriangleBlurring == true)
			{
				ApplyAdaptiveTriangleBlur(Camera.current, optimRenderTarget, resolvedFrameFreeView);
			}
			Graphics.Blit(resolvedFrameFreeView, cameraTarget);
		}
	}

	public void RenderProceduralPrimitivesOptimScene(Camera cameraToUse, ComputeBuffer[] primitiveBufferToUse, RenderTexture renderTargetToUse, RenderTexture idRenderTargetToUse, bool needsNewWorldAlphaSort = true)
	{
		// Camera setup
		Matrix4x4 cameraVP = GL.GetGPUProjectionMatrix(cameraToUse.projectionMatrix, true) * cameraToUse.worldToCameraMatrix;
		rasterMaterial.SetMatrix("_CameraMatrixVP", cameraVP);
		rasterMaterial.SetVector("_CurrentCameraWorldPos", cameraToUse.transform.position);
		int vertexCount = 3;

		// Clear resolve target
		primitiveRendererCS.SetTexture(kernelClearRenderTarget, "_ResolvedFrameRW", renderTargetToUse);
		primitiveRendererCS.Dispatch(kernelClearRenderTarget, (int)math.ceil(internalOptimResolution.x / 16.0f), (int)math.ceil(internalOptimResolution.y / 16.0f), 1);

		// Pre-blit env map to color resolve target // TODO : won't work with Sorted Alpha
		if (backgroundMode == BackgroundMode.EnvMap)
		{
			Matrix4x4 invCameraVP = cameraVP.inverse;
			envMapMaterial.SetInt("_EnvMapResolution", envMapResolution);
			envMapMaterial.SetMatrix("_CameraInvVP", invCameraVP);
			envMapMaterial.SetVector("_CurrentCameraWorldPos", cameraToUse.transform.position);
			envMapMaterial.SetBuffer("_EnvMapPrimitiveBuffer", primitiveBufferToUse[1]);
			envMapMaterial.SetFloat("_DoColorOrViewDir", 0.0f);
			Graphics.Blit(null, renderTargetToUse, envMapMaterial);
		}

		// Render primitives ID+Depth
		rasterMaterial.SetFloat("_SimpleColorRender", 0.0f);
		rasterMaterial.SetInt("_CurrentFrame", currentViewPoint);
		rasterMaterial.SetInt("_OutputWidth", internalOptimResolution.x);
		rasterMaterial.SetInt("_OutputHeight", internalOptimResolution.y);
		rasterMaterial.SetInt("_PrimitiveCount", primitiveBufferToUse[0].count);
		rasterMaterial.SetInt("_MutationsPerFrame", antitheticMutationsPerFrame);
		rasterMaterial.SetInt("_MaxFragmentsPerPixel", maxFragmentsPerPixel);
		rasterMaterial.SetBuffer("_PrimitiveBuffer", primitiveBufferToUse[0]);
		rasterMaterial.SetFloat("_UseSortedPrimitiveIDs", 0.0f);
		rasterMaterial.SetFloat("_DoVertexWelding", doStructuralWelding == true ? 1.0f : 0.0f);

		// Resolve render target color
		if (transparencyMode == TransparencyMode.None)
		{
			// Render
			Graphics.SetRenderTarget(idRenderTargetToUse.colorBuffer, idRenderTargetToUse.depthBuffer);
			GL.Clear(true, true, depthIDClearColor, 1.0f);
			if (backgroundMode == BackgroundMode.EnvMap)
			{
				envMapMaterial.SetFloat("_DoColorOrViewDir", 1.0f);
				Graphics.Blit(null, idRenderTargetToUse, envMapMaterial);
			}
			rasterMaterial.SetPass(0);
			Graphics.DrawProceduralNow(MeshTopology.Triangles, vertexCount, primitiveBufferToUse[0].count);

			// Resolve opaque render
			primitiveRendererCS.SetBuffer(kernelResolveOpaqueRender, "_PrimitiveBuffer", primitiveBufferToUse[0]);
			primitiveRendererCS.SetTexture(kernelResolveOpaqueRender, "_DepthIDBufferRW", idRenderTargetToUse);
			primitiveRendererCS.SetTexture(kernelResolveOpaqueRender, "_ResolvedFrameRW", renderTargetToUse);
			primitiveRendererCS.SetBuffer(kernelResolveOpaqueRender, "_PrimitiveKillCounters", primitiveKillCounters);
			primitiveRendererCS.Dispatch(kernelResolveOpaqueRender, (int)math.ceil(internalOptimResolution.x / 16.0f), (int)math.ceil(internalOptimResolution.y / 16.0f), 1);
		}
		/*else if (transparencyMode == TransparencyMode.SortedAlpha)
		{
			// Clear alpha stuff
			primitiveRendererCS.SetBuffer(kernelResetAlphaCounters, "_PerPixelFragmentCounter", perPixelFragmentCounterBuffer);
			primitiveRendererCS.SetBuffer(kernelResetAlphaCounters, "_PerPixelFragmentList", perPixelFragmentListBuffer);
			primitiveRendererCS.Dispatch(kernelResetAlphaCounters, (int)math.ceil(internalOptimResolution.x / 16.0f), (int)math.ceil(internalOptimResolution.y / 16.0f), 1);

			// Sort primitive render order based on distance to camera
			if (needsNewWorldAlphaSort == true)
			{
				primitiveRendererCS.SetMatrix("_SortMatrixMV", cameraToUse.worldToCameraMatrix);
				primitiveRendererCS.SetBuffer(kernelInitBitonicSortPrimitives, "_PrimitiveBuffer", primitiveBufferToUse[0]);
				primitiveRendererCS.SetBuffer(kernelInitBitonicSortPrimitives, "_SortedPrimitiveIDs", sortedPrimitiveIDBuffer);
				primitiveRendererCS.SetBuffer(kernelInitBitonicSortPrimitives, "_SortedPrimitiveDistances", sortedPrimitiveDistanceBuffer);
				DispatchCompute1D(primitiveRendererCS, kernelInitBitonicSortPrimitives, primitiveCount, 256);
				m_Sorter.Dispatch(m_SorterArgs);
			}

			// Render
			rasterMaterial.SetBuffer("_SortedPrimitiveIDs", sortedPrimitiveIDBuffer);
			rasterMaterial.SetFloat("_UseSortedPrimitiveIDs", 1.0f);
			rasterMaterial.SetPass(0);
			Graphics.SetRenderTarget(idRenderTargetToUse.colorBuffer, idRenderTargetToUse.depthBuffer);
			Graphics.SetRandomWriteTarget(4, perPixelFragmentCounterBuffer, true);
			Graphics.SetRandomWriteTarget(5, perPixelFragmentListBuffer, true);
			GL.Clear(true, true, Color.clear, 1.0f);
			Graphics.DrawProceduralNow(MeshTopology.Triangles, vertexCount, primitiveBufferToUse[0].count);

			// We use the render output directly in this mode (depth+ID is output in a different buffer)
			Graphics.Blit(idRenderTargetToUse, renderTargetToUse);

			// Count actual fragment count that contributed and reset kill counters
			primitiveRendererCS.SetFloat("_AlphaContributingCutoff", alphaContributingCutoff);
			primitiveRendererCS.SetBuffer(kernelFindContributingAlphaFragments, "_PrimitiveBuffer", primitiveBufferToUse[0]);
			primitiveRendererCS.SetBuffer(kernelFindContributingAlphaFragments, "_PerPixelFragmentCounter", perPixelFragmentCounterBuffer);
			primitiveRendererCS.SetBuffer(kernelFindContributingAlphaFragments, "_PerPixelFragmentList", perPixelFragmentListBuffer);
			primitiveRendererCS.SetBuffer(kernelFindContributingAlphaFragments, "_PrimitiveKillCounters", primitiveKillCounters);
			primitiveRendererCS.Dispatch(kernelFindContributingAlphaFragments, (int)math.ceil(internalOptimResolution.x / 16.0f), (int)math.ceil(internalOptimResolution.y / 16.0f), 1);
		}
		else if (transparencyMode == TransparencyMode.StochasticAlpha)
		{
			// Clear alpha stuff
			primitiveRendererCS.SetBuffer(kernelResetAlphaCounters, "_PerPixelFragmentCounter", perPixelFragmentCounterBuffer);
			primitiveRendererCS.SetBuffer(kernelResetAlphaCounters, "_PerPixelFragmentList", perPixelFragmentListBuffer);
			primitiveRendererCS.Dispatch(kernelResetAlphaCounters, (int)math.ceil(internalOptimResolution.x / 16.0f), (int)math.ceil(internalOptimResolution.y / 16.0f), 1);

			for (int i = 0; i < maxFragmentsPerPixel; i++)
			{
				// Render
				rasterMaterial.SetPass(0);
				rasterMaterial.SetInt("_CurrentStochasticFrame", currentStochasticFrame++);
				Graphics.SetRenderTarget(idRenderTargetToUse.colorBuffer, idRenderTargetToUse.depthBuffer);
				GL.Clear(true, true, depthIDClearColor, 1.0f);
				Graphics.DrawProceduralNow(MeshTopology.Triangles, vertexCount, primitiveBufferToUse[0].count);

				// Resolve opaque render (additive)
				primitiveRendererCS.SetInt("_CurrentStochasticFrame", i);
				primitiveRendererCS.SetBuffer(kernelResolveOpaqueRender, "_PrimitiveBuffer", primitiveBufferToUse[0]);
				primitiveRendererCS.SetTexture(kernelResolveOpaqueRender, "_DepthIDBufferRW", idRenderTargetToUse);
				primitiveRendererCS.SetTexture(kernelResolveOpaqueRender, "_ResolvedFrameRW", renderTargetToUse);
				primitiveRendererCS.SetBuffer(kernelResolveOpaqueRender, "_PrimitiveKillCounters", primitiveKillCounters);
				primitiveRendererCS.SetBuffer(kernelResolveOpaqueRender, "_PerPixelFragmentList", perPixelFragmentListBuffer);
				primitiveRendererCS.Dispatch(kernelResolveOpaqueRender, (int)math.ceil(internalOptimResolution.x / 16.0f), (int)math.ceil(internalOptimResolution.y / 16.0f), 1);
			}
		}*/

		if (optimAdaptiveTriangleBlurring == true)
		{
			ApplyAdaptiveTriangleBlur(cameraToUse, idRenderTargetToUse, renderTargetToUse);
		}
	}

	public void BlurImageBloomStyle(RenderTexture target)
	{
		RenderTexture source = RenderTexture.GetTemporary(target.width, target.height, 0, target.format);
		Graphics.Blit(target, source);

		int iterations = 5;
		int width = source.width / 2;
		int height = source.height / 2;
		RenderTextureFormat format = source.format;

		RenderTexture[] textures = new RenderTexture[16];
		RenderTexture currentDestination = textures[0] = RenderTexture.GetTemporary(width, height, 0, format);
		Graphics.Blit(source, currentDestination, blurringMaterial, 0);
		RenderTexture currentSource = currentDestination;

		int i = 1;
		for (; i < iterations; i++)
		{
			width /= 2;
			height /= 2;
			if (width < 2 || height < 2)
				break;
			currentDestination = textures[i] = RenderTexture.GetTemporary(width, height, 0, format);
			Graphics.Blit(currentSource, currentDestination, blurringMaterial, 0);
			currentSource = currentDestination;
		}

		for (i -= 2; i >= 0; i--)
		{
			currentDestination = textures[i];
			textures[i] = null;
			Graphics.Blit(currentSource, currentDestination, blurringMaterial, 1);
			RenderTexture.ReleaseTemporary(currentSource);
			currentSource = currentDestination;
		}

		Graphics.Blit(currentSource, target, blurringMaterial, 1);
		RenderTexture.ReleaseTemporary(currentSource);
		RenderTexture.ReleaseTemporary(source);
	}

	public void ApplyAdaptiveTriangleBlur(Camera cameraToUse, RenderTexture idBuffer, RenderTexture target)
	{
		Graphics.Blit(target, tempResolvedFrameBuffer);
		tempResolvedFrameBuffer.GenerateMips();
		adaptiveTriangleBlurMaterial.SetBuffer("_PrimitiveBuffer", primitiveBuffer[0]);
		adaptiveTriangleBlurMaterial.SetInt("_PrimitiveCount", primitiveBuffer[0].count);
		adaptiveTriangleBlurMaterial.SetTexture("_DepthIDBuffer", idBuffer);
		adaptiveTriangleBlurMaterial.SetVector("_CustomWorldSpaceCameraPos", cameraToUse.transform.position);
		adaptiveTriangleBlurMaterial.SetFloat("_CameraFovVRad", cameraToUse.fieldOfView * Mathf.Deg2Rad);
		adaptiveTriangleBlurMaterial.SetMatrix("_CameraMatrixVP", GL.GetGPUProjectionMatrix(cameraToUse.projectionMatrix, true) * cameraToUse.worldToCameraMatrix);
		Graphics.Blit(tempResolvedFrameBuffer, target, adaptiveTriangleBlurMaterial);
	}




	// ======================= OPTIMIZATION =======================
	public void UpdateSchedulingStuff()
	{
		// Handle live doubling of primitive count
		if (needsToDoublePrimitives == true || (primitiveDoublingCounter < primitiveDoublingCountAndInterval.x && currentOptimStep > 1 && currentOptimStep % primitiveDoublingCountAndInterval.y == 0))
		{
			needsToDoublePrimitives = false;
			if (doStructuralLoss == false)
				DoublePrimitiveCountBySubdivision();
			else
				TriplePrimitiveCountBySubdivision();
			//DoublePrimitiveCountByNewInsertion();
			setPrimitiveCount = primitiveCount;
			primitiveDoublingCounter++;
		}

		// Handle live doubling of optim resolution
		if (resolutionDoublingCounter < resolutionDoublingCountAndInterval.x && currentOptimStep > 1 && currentOptimStep % resolutionDoublingCountAndInterval.y == 0)
		{
			optimResolutionFactor *= 2;
			resolutionDoublingCounter++;
		}

		// Position learning rate decay
		if (lrGeometryStartMulAndSpeed.x != 0 && lrGeometryStartMulAndSpeed.y != 0)
		{
			learningRatePosition = math.exp(-currentOptimStep * lrGeometryStartMulAndSpeed.y) * (learningRatePositionStartEnd.x - learningRatePositionStartEnd.y) + learningRatePositionStartEnd.y;
		}

		if (lrGlobalStartMulAndSpeed.x != 0 && lrGlobalStartMulAndSpeed.y != 0)
		{
			globalLearningRate = math.exp(-currentOptimStep * lrGlobalStartMulAndSpeed.y) * (learningRateGlobalStartEnd.x - learningRateGlobalStartEnd.y) + learningRateGlobalStartEnd.y;
		}
	}

	public void SetOptimizerLearningRatesAndGetSeparateCount(int currentParameterGroup, bool dontDoSeparation = false)
	{
		// Normal mode
		float learningRateModifier = globalLearningRate;// * (1.0f / optimSupersampling);
		if (parameterSeparationMode == ParameterOptimSeparationMode.None || dontDoSeparation == true)
		{
			mutationOptimizerCS.SetFloat("_LearningRatePosition", learningRatePosition * learningRateModifier);
			mutationOptimizerCS.SetFloat("_LearningRateColor", learningRateColor * learningRateModifier);
			mutationOptimizerCS.SetFloat("_LearningRateAlpha", learningRateAlpha * learningRateModifier);
			mutationOptimizerCS.SetFloat("_LearningRateEnvMap", learningRateEnvMap * learningRateModifier);
			if (parameterSeparationMode == ParameterOptimSeparationMode.None)
				optimStepsSeparateCount = 1;
			return;
		}

		// Disable all learning rates
		mutationOptimizerCS.SetFloat("_LearningRatePosition", 0.0f);
		mutationOptimizerCS.SetFloat("_LearningRateRotation", 0.0f);
		mutationOptimizerCS.SetFloat("_LearningRateOffsets", 0.0f);
		mutationOptimizerCS.SetFloat("_LearningRateColor", 0.0f);
		mutationOptimizerCS.SetFloat("_LearningRateAlpha", 0.0f);
		mutationOptimizerCS.SetFloat("_LearningRateEnvMap", 0.0f);

		// GEOMETRY/APPEARANCE SEPARATION
		if (parameterSeparationMode == ParameterOptimSeparationMode.GeometryAndAppearance)
		{
			optimStepsSeparateCount = 2;
			if (optimPrimitive == PrimitiveType.TrianglesSolidUnlit || optimPrimitive == PrimitiveType.TrianglesGradientUnlit || optimPrimitive == PrimitiveType.TrianglesGaussianUnlit)
			{
				if (currentParameterGroup % 2 == 0)
				{
					mutationOptimizerCS.SetFloat("_LearningRatePosition", learningRatePosition * learningRateModifier);
				}
				else
				{
					mutationOptimizerCS.SetFloat("_LearningRateColor", learningRateColor * learningRateModifier);
					if (transparencyMode != TransparencyMode.None)
						mutationOptimizerCS.SetFloat("_LearningRateAlpha", learningRateAlpha * learningRateModifier);
					if (backgroundMode == BackgroundMode.EnvMap)
						mutationOptimizerCS.SetFloat("_LearningRateEnvMap", learningRateEnvMap * learningRateModifier);
				}
			}
		}
		// FULL SEPARATION
		else if (parameterSeparationMode == ParameterOptimSeparationMode.Full)
		{
			if (optimPrimitive == PrimitiveType.TrianglesSolidUnlit || optimPrimitive == PrimitiveType.TrianglesGradientUnlit || optimPrimitive == PrimitiveType.TrianglesGaussianUnlit)
			{
				List<Tuple<string, float>> usedParams = new List<Tuple<string, float>>();
				if (learningRatePosition > 0.0f) usedParams.Add(new Tuple<string, float>("_LearningRatePosition", learningRatePosition));
				if (learningRateColor > 0.0f) usedParams.Add(new Tuple<string, float>("_LearningRateColor", learningRateColor));
				if (transparencyMode != TransparencyMode.None && learningRateAlpha > 0.0f) usedParams.Add(new Tuple<string, float>("_LearningRateAlpha", learningRateAlpha));
				if (backgroundMode == BackgroundMode.EnvMap && learningRateEnvMap > 0.0f) usedParams.Add(new Tuple<string, float>("_LearningRateEnvMap", learningRateEnvMap));
				int currentParam = currentParameterGroup % usedParams.Count;
				mutationOptimizerCS.SetFloat(usedParams[currentParam].Item1, usedParams[currentParam].Item2 * learningRateModifier);
				optimStepsSeparateCount = usedParams.Count;
			}
		}
	}

	public void ResetOptimizationStep(int primitiveGroupToUse)
	{
		int kernelToUse = primitiveGroupToUse == 0 ? kernelResetMutationLossAccumulation : kernelEnvMapResetMutationLossAccumulation;
		string suffix = primitiveGroupToUse == 0 ? "" : "Float3";
		mutationOptimizerCS.SetBuffer(kernelToUse, "_PrimitiveGradientsOptimStep" + suffix, optimStepGradientsBuffer[primitiveGroupToUse]);
		mutationOptimizerCS.SetBuffer(kernelToUse, "_PrimitiveMutationError", optimStepMutationError[primitiveGroupToUse]);
		DispatchCompute1D(mutationOptimizerCS, kernelToUse, primitiveBuffer[primitiveGroupToUse].count, 256);
	}

	public void AccumulateOptimizationStep(int primitiveGroupToUse, int currentParameterGroup)
	{
		// Accumulate per pixel image loss
		int kernelToUse = primitiveGroupToUse == 0 ? kernelAccumulateMutationLoss : kernelEnvMapAccumulateMutationLoss;
		string suffix = primitiveGroupToUse == 0 ? "" : "Float3";
		mutationOptimizerCS.SetTexture(kernelToUse, "_TargetTexture", targetFrameBuffer);
		mutationOptimizerCS.SetTexture(kernelToUse, "_ResolvedFrameMutatedMinus", resolvedFrameMutatedMinus);
		mutationOptimizerCS.SetTexture(kernelToUse, "_ResolvedFrameMutatedPlus", resolvedFrameMutatedPlus);
		mutationOptimizerCS.SetBuffer(kernelToUse, "_PrimitiveMutationError", optimStepMutationError[primitiveGroupToUse]);
		mutationOptimizerCS.SetBuffer(kernelToUse, "_StructuralVertexWeldingBuffer", structuralVertexWeldingBuffer);
		if (transparencyMode == TransparencyMode.None)
		{
			mutationOptimizerCS.SetTexture(kernelToUse, "_DepthIDBufferMutatedMinus", optimRenderTargetMutatedMinus);
			mutationOptimizerCS.SetTexture(kernelToUse, "_DepthIDBufferMutatedPlus", optimRenderTarget);
		}
		else
		{
			mutationOptimizerCS.SetBuffer(kernelToUse, "_PerPixelFragmentCounter", perPixelFragmentCounterBuffer);
			mutationOptimizerCS.SetBuffer(kernelToUse, "_PerPixelFragmentList", perPixelFragmentListBuffer);
		}
		mutationOptimizerCS.Dispatch(kernelToUse, (int)math.ceil(internalOptimResolution.x / 16.0f), (int)math.ceil(internalOptimResolution.y / 16.0f), 1);

		// Accumulate structural distance loss (only when optimizing positions !)
		if ((doStructuralLoss == true || doStructuralWelding == true) && currentParameterGroup == 0 && primitiveGroupToUse == 0 && transparencyMode == TransparencyMode.None)
		{
			// Update closest edge pairings
			int kernelToUse3 = kernelAccumulateMutationLossStructural;
			mutationOptimizerCS.SetBuffer(kernelToUse3, "_PrimitiveBuffer", primitiveBuffer[primitiveGroupToUse]);
			mutationOptimizerCS.SetBuffer(kernelToUse3, "_PrimitiveBufferMutated", primitiveBufferMutated[primitiveGroupToUse]);
			mutationOptimizerCS.SetBuffer(kernelToUse3, "_PrimitiveMutationError", optimStepMutationError[primitiveGroupToUse]);
			mutationOptimizerCS.SetTexture(kernelToUse3, "_DepthIDBufferMutatedMinus", optimRenderTargetMutatedMinus);
			mutationOptimizerCS.SetTexture(kernelToUse3, "_DepthIDBufferMutatedPlus", optimRenderTarget);
			mutationOptimizerCS.SetBuffer(kernelToUse3, "_StructuralEdgeClosestNeighbourBuffer", structuralEdgeClosestNeighbourBuffer);
			mutationOptimizerCS.SetBuffer(kernelToUse3, "_StructuralVertexWeldingBuffer", structuralVertexWeldingBuffer);
			mutationOptimizerCS.Dispatch(kernelToUse3, (int)math.ceil(internalOptimResolution.x / 16.0f), (int)math.ceil(internalOptimResolution.y / 16.0f), 1);
		}

		// Accumulate gradients
		int kernelToUse2 = primitiveGroupToUse == 0 ? kernelAccumulateMutationGradientsResetLoss : kernelEnvMapAccumulateMutationGradientsResetLoss;
		mutationOptimizerCS.SetBuffer(kernelToUse2, "_PrimitiveBuffer" + suffix, primitiveBuffer[primitiveGroupToUse]);
		mutationOptimizerCS.SetBuffer(kernelToUse2, "_PrimitiveBufferMutated" + suffix, primitiveBufferMutated[primitiveGroupToUse]);
		mutationOptimizerCS.SetBuffer(kernelToUse2, "_PrimitiveMutationError", optimStepMutationError[primitiveGroupToUse]);
		mutationOptimizerCS.SetBuffer(kernelToUse2, "_PrimitiveGradientsOptimStep" + suffix, optimStepGradientsBuffer[primitiveGroupToUse]);
		DispatchCompute1D(mutationOptimizerCS, kernelToUse2, primitiveBuffer[primitiveGroupToUse].count, 256);
	}

	public void ApplyOptimizationStep(int primitiveGroupToUse)
	{
		int kernelToUse = primitiveGroupToUse == 0 ? kernelApplyRandomMutation : kernelEnvMapApplyRandomMutation;
		string suffix = primitiveGroupToUse == 0 ? "" : "Float3";
		mutationOptimizerCS.SetBuffer(kernelToUse, "_PrimitiveBuffer" + suffix, primitiveBuffer[primitiveGroupToUse]);
		mutationOptimizerCS.SetBuffer(kernelToUse, "_PrimitiveGradientsMoments1" + suffix, gradientMoments1Buffer[primitiveGroupToUse]);
		mutationOptimizerCS.SetBuffer(kernelToUse, "_PrimitiveGradientsMoments2" + suffix, gradientMoments2Buffer[primitiveGroupToUse]);
		mutationOptimizerCS.SetBuffer(kernelToUse, "_PrimitiveGradientsOptimStep" + suffix, optimStepGradientsBuffer[primitiveGroupToUse]);
		mutationOptimizerCS.SetBuffer(kernelToUse, "_PrimitiveOptimStepCounter", optimStepCounterBuffer[primitiveGroupToUse]);
		mutationOptimizerCS.SetBuffer(kernelToUse, "_PrimitiveMutationError", optimStepMutationError[primitiveGroupToUse]);
		DispatchCompute1D(mutationOptimizerCS, kernelToUse, primitiveBuffer[primitiveGroupToUse].count, 256);

		// Apply vertex welding indirection
		if (doStructuralWelding == true && primitiveGroupToUse == 0)
			ApplyVertexWeldingIndirection(primitiveBuffer[0]);
	}

	public void CreateNewRandomMutation(int primitiveGroupToUse)
	{
		int kernelToUse = primitiveGroupToUse == 0 ? kernelCreateNewRandomMutation : kernelEnvMapCreateNewRandomMutation;
		string suffix = primitiveGroupToUse == 0 ? "" : "Float3";
		mutationOptimizerCS.SetVector("_WorldSpaceCameraPos", cameraOptim.transform.position);
		mutationOptimizerCS.SetFloat("_CameraFovVRad", cameraOptim.fieldOfView * Mathf.Deg2Rad);
		mutationOptimizerCS.SetBuffer(kernelToUse, "_PrimitiveBuffer" + suffix, primitiveBuffer[primitiveGroupToUse]);
		mutationOptimizerCS.SetBuffer(kernelToUse, "_PrimitiveBufferMutated" + suffix, primitiveBufferMutated[primitiveGroupToUse]);
		DispatchCompute1D(mutationOptimizerCS, kernelToUse, primitiveBuffer[primitiveGroupToUse].count, 256);

		// Apply vertex welding indirection
		if (doStructuralWelding == true && primitiveGroupToUse == 0)
			ApplyVertexWeldingIndirection(primitiveBufferMutated[0]);
	}

	public void PerformPrimitiveResampling(int primitiveGroupToUse)
	{
		if (doPrimitiveResampling != setPrimitiveResampling)
		{
			// Re-init visibility counters
			mutationOptimizerCS.SetInt("_PrimitiveCount", primitiveBuffer[0].count);
			mutationOptimizerCS.SetBuffer(kernelResetVisibilityCounter, "_PrimitiveKillCounters", primitiveKillCounters);
			DispatchCompute1D(mutationOptimizerCS, kernelResetVisibilityCounter, primitiveBuffer[primitiveGroupToUse].count, 256);
			setPrimitiveResampling = doPrimitiveResampling;
		}

		if (doPrimitiveResampling == false)
			return;

		// Only perform resampling at desired interval
		if (currentOptimStep % resamplingInterval != 0)
			return;

		// List valid and invalid primitive IDs
		appendValidIDsBuffer.SetCounterValue(0);
		appendInvalidIDsBuffer.SetCounterValue(0);
		mutationOptimizerCS.SetInt("_CurrentPairingOffset", (int)(UnityEngine.Random.value * primitiveCount));
		mutationOptimizerCS.SetBuffer(kernelListValidAndInvalidPrimitiveIDs, "_PrimitiveBuffer", primitiveBuffer[0]);
		mutationOptimizerCS.SetBuffer(kernelListValidAndInvalidPrimitiveIDs, "_PrimitiveGradientsMoments1", gradientMoments1Buffer[0]);
		mutationOptimizerCS.SetBuffer(kernelListValidAndInvalidPrimitiveIDs, "_PrimitiveGradientsMoments2", gradientMoments2Buffer[0]);
		mutationOptimizerCS.SetBuffer(kernelListValidAndInvalidPrimitiveIDs, "_PrimitiveOptimStepCounter", optimStepCounterBuffer[0]);
		mutationOptimizerCS.SetBuffer(kernelListValidAndInvalidPrimitiveIDs, "_PrimitiveKillCounters", primitiveKillCounters);
		mutationOptimizerCS.SetBuffer(kernelListValidAndInvalidPrimitiveIDs, "_AppendValidPrimitiveIDs", appendValidIDsBuffer);
		mutationOptimizerCS.SetBuffer(kernelListValidAndInvalidPrimitiveIDs, "_AppendInvalidPrimitiveIDs", appendInvalidIDsBuffer);
		DispatchCompute1D(mutationOptimizerCS, kernelListValidAndInvalidPrimitiveIDs, primitiveCount, 256);

		// Init resampling indirect dispatch args
		ComputeBuffer.CopyCount(appendValidIDsBuffer, argsValidIDsBuffer, 0);
		ComputeBuffer.CopyCount(appendInvalidIDsBuffer, argsInvalidIDsBuffer, 0);
		mutationOptimizerCS.SetBuffer(kernelInitArgsResampling, "_ArgsValidPrimitiveIDs", argsValidIDsBuffer);
		mutationOptimizerCS.SetBuffer(kernelInitArgsResampling, "_ArgsInvalidPrimitiveIDs", argsInvalidIDsBuffer);
		mutationOptimizerCS.SetBuffer(kernelInitArgsResampling, "_ArgsResampling", argsResampling);
		mutationOptimizerCS.Dispatch(kernelInitArgsResampling, 1, 1, 1);

		// Sort valid primitives by importance criteria
		mutationOptimizerCS.SetInt("_PrimitiveCountPow2", sortedValidPrimitiveIDBuffer.count);
		mutationOptimizerCS.SetBuffer(kernelInitBitonicSortValidPrimitives, "_PrimitiveBuffer", primitiveBuffer[0]);
		mutationOptimizerCS.SetBuffer(kernelInitBitonicSortValidPrimitives, "_PrimitiveGradientsMoments1", gradientMoments1Buffer[0]);
		mutationOptimizerCS.SetBuffer(kernelInitBitonicSortValidPrimitives, "_PrimitiveGradientsMoments2", gradientMoments2Buffer[0]);
		mutationOptimizerCS.SetBuffer(kernelInitBitonicSortValidPrimitives, "_ReadValidPrimitiveIDs", appendValidIDsBuffer);
		mutationOptimizerCS.SetBuffer(kernelInitBitonicSortValidPrimitives, "_SortedValidPrimitiveIDs", sortedValidPrimitiveIDBuffer);
		mutationOptimizerCS.SetBuffer(kernelInitBitonicSortValidPrimitives, "_ArgsValidPrimitiveIDs", argsValidIDsBuffer);
		DispatchCompute1D(mutationOptimizerCS, kernelInitBitonicSortValidPrimitives, primitiveCount, 256);
		mutationOptimizerCS.SetBuffer(kernelBitonicSortValidPrimitives, "_SortedValidPrimitiveIDs", sortedValidPrimitiveIDBuffer);
		for (uint d2 = 1; d2 < sortedValidPrimitiveIDBuffer.count; d2 *= 2)
		{
			for (uint d1 = d2; d1 >= 1u; d1 /= 2)
			{
				mutationOptimizerCS.SetInt("_SortLoopValueX", (int)d1);
				mutationOptimizerCS.SetInt("_SortLoopValueY", (int)d2);
				DispatchCompute1D(mutationOptimizerCS, kernelBitonicSortValidPrimitives, sortedValidPrimitiveIDBuffer.count, 256);
			}
		}

		// Resample primitive pairs
		if (doStructuralLoss == false)
		{
			mutationOptimizerCS.SetInt("_CurrentFrame", currentViewPoint);
			mutationOptimizerCS.SetBuffer(kernelPairResampling, "_PrimitiveBuffer", primitiveBuffer[0]);
			mutationOptimizerCS.SetBuffer(kernelPairResampling, "_PrimitiveKillCounters", primitiveKillCounters);
			mutationOptimizerCS.SetBuffer(kernelPairResampling, "_PrimitiveGradientsMoments1", gradientMoments1Buffer[0]);
			mutationOptimizerCS.SetBuffer(kernelPairResampling, "_PrimitiveGradientsMoments2", gradientMoments2Buffer[0]);
			mutationOptimizerCS.SetBuffer(kernelPairResampling, "_PrimitiveOptimStepCounter", optimStepCounterBuffer[0]);
			mutationOptimizerCS.SetBuffer(kernelPairResampling, "_ArgsResampling", argsResampling);
			mutationOptimizerCS.SetBuffer(kernelPairResampling, "_ReadValidPrimitiveIDs", appendValidIDsBuffer);
			mutationOptimizerCS.SetBuffer(kernelPairResampling, "_SortedValidPrimitiveIDs", sortedValidPrimitiveIDBuffer);
			mutationOptimizerCS.SetBuffer(kernelPairResampling, "_ReadInvalidPrimitiveIDs", appendInvalidIDsBuffer);
			mutationOptimizerCS.DispatchIndirect(kernelPairResampling, argsResampling);
		}
		// Resample primitive triples
		else
		{
			mutationOptimizerCS.SetInt("_CurrentFrame", currentViewPoint);
			mutationOptimizerCS.SetBuffer(kernelTripleResampling, "_PrimitiveBuffer", primitiveBuffer[0]);
			mutationOptimizerCS.SetBuffer(kernelTripleResampling, "_PrimitiveKillCounters", primitiveKillCounters);
			mutationOptimizerCS.SetBuffer(kernelTripleResampling, "_PrimitiveGradientsMoments1", gradientMoments1Buffer[0]);
			mutationOptimizerCS.SetBuffer(kernelTripleResampling, "_PrimitiveGradientsMoments2", gradientMoments2Buffer[0]);
			mutationOptimizerCS.SetBuffer(kernelTripleResampling, "_PrimitiveOptimStepCounter", optimStepCounterBuffer[0]);
			mutationOptimizerCS.SetBuffer(kernelTripleResampling, "_ArgsResampling", argsResampling);
			mutationOptimizerCS.SetBuffer(kernelTripleResampling, "_ReadValidPrimitiveIDs", appendValidIDsBuffer);
			mutationOptimizerCS.SetBuffer(kernelTripleResampling, "_SortedValidPrimitiveIDs", sortedValidPrimitiveIDBuffer);
			mutationOptimizerCS.SetBuffer(kernelTripleResampling, "_ReadInvalidPrimitiveIDs", appendInvalidIDsBuffer);
			mutationOptimizerCS.DispatchIndirect(kernelTripleResampling, argsResampling);
		}

		//int[] args0 = new int[4];
		//argsValidIDsBuffer.GetData(args0);
		//int[] args1 = new int[4];
		//argsInvalidIDsBuffer.GetData(args1);
		//int[] args2 = new int[4];
		//argsResampling.GetData(args2);
		//int x = 0;
	}

	public void DoublePrimitiveCountBySubdivision()
	{
		if (optimPrimitive != PrimitiveType.TrianglesSolidUnlit && optimPrimitive != PrimitiveType.TrianglesGradientUnlit && optimPrimitive != PrimitiveType.TrianglesGaussianUnlit)
			return;

		// Double all the buffer sizes and copy existing primitives to first half
		primitiveCount = primitiveCount * 2;
		int primitiveByteSize = GetPrimitiveFloatSize(optimPrimitive) * 4;
		byte[] tempDataCopy = new byte[primitiveCount / 2 * primitiveByteSize];
		primitiveBuffer[0].GetData(tempDataCopy);
		primitiveBuffer[0].Release();
		primitiveBuffer[0] = new ComputeBuffer(primitiveCount, primitiveByteSize);
		primitiveBuffer[0].SetData(tempDataCopy);
		InitAllOptimBuffers();

		// Hijack pair resampling code by creating fake invalid primitive and sorted valid primitive buffers
		uint2[] sortedValidIDs = new uint2[primitiveCount / 2];
		for (int i = 0; i < sortedValidIDs.Length; i++)
			sortedValidIDs[i] = new uint2((uint)i, 0);
		uint[] invalidIDs = new uint[primitiveCount / 2];
		for (int i = 0; i < invalidIDs.Length; i++)
			invalidIDs[i] = (uint)(primitiveCount / 2 + i);
		ComputeBuffer tempSortedValidBuffer = new ComputeBuffer(primitiveCount / 2, sizeof(uint) * 2);
		tempSortedValidBuffer.SetData(sortedValidIDs);
		ComputeBuffer tempInvalidBuffer = new ComputeBuffer(primitiveCount / 2, sizeof(uint));
		tempInvalidBuffer.SetData(invalidIDs);

		// Dispatch args
		int[] temp = new int[4];
		int threadGroupCount = (int)math.ceil(primitiveCount / 2 / 256.0);
		if (threadGroupCount < 65536)
		{
			temp[0] = threadGroupCount;
			temp[1] = 1;
			temp[2] = 1;
			temp[3] = primitiveCount / 2;
		}
		else
		{
			int threadGroupCountDistributed = (int)math.ceil(math.pow(threadGroupCount, 1.0f / 2.0f));
			temp[0] = threadGroupCountDistributed;
			temp[1] = threadGroupCountDistributed;
			temp[2] = 1;
			temp[3] = primitiveCount / 2;
		}
		argsResampling.SetData(temp);

		// Run resampling
		mutationOptimizerCS.SetInt("_CurrentFrame", currentViewPoint);
		mutationOptimizerCS.SetBuffer(kernelPairResampling, "_PrimitiveBuffer", primitiveBuffer[0]);
		mutationOptimizerCS.SetBuffer(kernelPairResampling, "_PrimitiveKillCounters", primitiveKillCounters);
		mutationOptimizerCS.SetBuffer(kernelPairResampling, "_PrimitiveGradientsMoments1", gradientMoments1Buffer[0]);
		mutationOptimizerCS.SetBuffer(kernelPairResampling, "_PrimitiveGradientsMoments2", gradientMoments2Buffer[0]);
		mutationOptimizerCS.SetBuffer(kernelPairResampling, "_PrimitiveOptimStepCounter", optimStepCounterBuffer[0]);
		mutationOptimizerCS.SetBuffer(kernelPairResampling, "_ArgsResampling", argsResampling);
		mutationOptimizerCS.SetBuffer(kernelPairResampling, "_SortedValidPrimitiveIDs", tempSortedValidBuffer);
		mutationOptimizerCS.SetBuffer(kernelPairResampling, "_ReadInvalidPrimitiveIDs", tempInvalidBuffer);
		mutationOptimizerCS.DispatchIndirect(kernelPairResampling, argsResampling);

		// Finish
		ResetKeywords(primitiveRendererCS, true, true, true);
		ResetKeywords(mutationOptimizerCS, true, true, true);
		ResetKeywords(rasterMaterial, true, true, true);
		ResetKeywords(adaptiveTriangleBlurMaterial, true, true, true);
		tempSortedValidBuffer.Release();
		tempInvalidBuffer.Release();
	}

	public void TriplePrimitiveCountBySubdivision()
	{
		if (optimPrimitive != PrimitiveType.TrianglesSolidUnlit && optimPrimitive != PrimitiveType.TrianglesGradientUnlit && optimPrimitive != PrimitiveType.TrianglesGaussianUnlit)
			return;

		// Triple all the buffer sizes and copy existing primitives to first third
		primitiveCount = primitiveCount * 3;
		int primitiveByteSize = GetPrimitiveFloatSize(optimPrimitive) * 4;
		byte[] tempDataCopy = new byte[primitiveCount / 3 * primitiveByteSize];
		primitiveBuffer[0].GetData(tempDataCopy);
		primitiveBuffer[0].Release();
		primitiveBuffer[0] = new ComputeBuffer(primitiveCount, primitiveByteSize);
		primitiveBuffer[0].SetData(tempDataCopy);
		InitAllOptimBuffers();

		// Hijack triple resampling code by creating fake invalid primitive and sorted valid primitive buffers
		uint2[] sortedValidIDs = new uint2[primitiveCount / 3];
		for (int i = 0; i < sortedValidIDs.Length; i++)
			sortedValidIDs[i] = new uint2((uint)i, 0);
		uint[] invalidIDs = new uint[(primitiveCount / 3) * 2];
		for (int i = 0; i < invalidIDs.Length; i++)
			invalidIDs[i] = (uint)(primitiveCount / 3 + i);
		ComputeBuffer tempSortedValidBuffer = new ComputeBuffer(primitiveCount / 3, sizeof(uint) * 2);
		tempSortedValidBuffer.SetData(sortedValidIDs);
		ComputeBuffer tempInvalidBuffer = new ComputeBuffer((primitiveCount / 3) * 2, sizeof(uint));
		tempInvalidBuffer.SetData(invalidIDs);

		// Dispatch args
		int[] temp = new int[4];
		int threadGroupCount = (int)math.ceil(primitiveCount / 3 / 256.0);
		if (threadGroupCount < 65536)
		{
			temp[0] = threadGroupCount;
			temp[1] = 1;
			temp[2] = 1;
			temp[3] = primitiveCount / 3;
		}
		else
		{
			int threadGroupCountDistributed = (int)math.ceil(math.pow(threadGroupCount, 1.0f / 2.0f));
			temp[0] = threadGroupCountDistributed;
			temp[1] = threadGroupCountDistributed;
			temp[2] = 1;
			temp[3] = primitiveCount / 3;
		}
		argsResampling.SetData(temp);

		// Run resampling
		mutationOptimizerCS.SetInt("_CurrentFrame", currentViewPoint);
		mutationOptimizerCS.SetBuffer(kernelTripleResampling, "_PrimitiveBuffer", primitiveBuffer[0]);
		mutationOptimizerCS.SetBuffer(kernelTripleResampling, "_PrimitiveKillCounters", primitiveKillCounters);
		mutationOptimizerCS.SetBuffer(kernelTripleResampling, "_PrimitiveGradientsMoments1", gradientMoments1Buffer[0]);
		mutationOptimizerCS.SetBuffer(kernelTripleResampling, "_PrimitiveGradientsMoments2", gradientMoments2Buffer[0]);
		mutationOptimizerCS.SetBuffer(kernelTripleResampling, "_PrimitiveOptimStepCounter", optimStepCounterBuffer[0]);
		mutationOptimizerCS.SetBuffer(kernelTripleResampling, "_ArgsResampling", argsResampling);
		mutationOptimizerCS.SetBuffer(kernelTripleResampling, "_SortedValidPrimitiveIDs", tempSortedValidBuffer);
		mutationOptimizerCS.SetBuffer(kernelTripleResampling, "_ReadInvalidPrimitiveIDs", tempInvalidBuffer);
		mutationOptimizerCS.DispatchIndirect(kernelTripleResampling, argsResampling);

		// Finish
		ResetKeywords(primitiveRendererCS, true, true, true);
		ResetKeywords(mutationOptimizerCS, true, true, true);
		ResetKeywords(rasterMaterial, true, true, true);
		ResetKeywords(adaptiveTriangleBlurMaterial, true, true, true);
		tempSortedValidBuffer.Release();
		tempInvalidBuffer.Release();
	}

	public void DoublePrimitiveCountByNewInsertion()
	{
		if (optimPrimitive != PrimitiveType.TrianglesSolidUnlit && optimPrimitive != PrimitiveType.TrianglesGradientUnlit && optimPrimitive != PrimitiveType.TrianglesGaussianUnlit)
			return;

		// Copy existing persistent data
		int primitiveByteSize = GetPrimitiveFloatSize(optimPrimitive) * 4;
		byte[] primitiveBufferCopy = new byte[primitiveCount * primitiveByteSize];
		primitiveBuffer[0].GetData(primitiveBufferCopy);
		byte[] gradientMoments1BufferCopy = new byte[primitiveCount * primitiveByteSize];
		gradientMoments1Buffer[0].GetData(gradientMoments1BufferCopy);
		byte[] gradientMoments2BufferCopy = new byte[primitiveCount * primitiveByteSize];
		gradientMoments2Buffer[0].GetData(gradientMoments2BufferCopy);
		byte[] optimStepCounterBufferCopy = new byte[primitiveCount * sizeof(int)];
		optimStepCounterBuffer[0].GetData(optimStepCounterBufferCopy);
		byte[] primitiveKillCountersCopy = new byte[primitiveCount * sizeof(int)];
		primitiveKillCounters.GetData(primitiveKillCountersCopy);

		// Double all the buffer sizes and reinit
		primitiveCount = primitiveCount * 2;
		primitiveBuffer[0].Release();
		InitPrimitiveBuffer();
		InitAllOptimBuffers();

		// Copy existing data to first half of new buffers
		primitiveBuffer[0].SetData(primitiveBufferCopy, 0, 0, primitiveBufferCopy.Length);
		gradientMoments1Buffer[0].SetData(gradientMoments1BufferCopy, 0, 0, gradientMoments1BufferCopy.Length);
		gradientMoments2Buffer[0].SetData(gradientMoments2BufferCopy, 0, 0, gradientMoments2BufferCopy.Length);
		optimStepCounterBuffer[0].SetData(optimStepCounterBufferCopy, 0, 0, optimStepCounterBufferCopy.Length);
		primitiveKillCounters.SetData(primitiveKillCountersCopy, 0, 0, primitiveKillCountersCopy.Length);

		// Finish
		ResetKeywords(primitiveRendererCS, true, true, true);
		ResetKeywords(mutationOptimizerCS, true, true, true);
		ResetKeywords(rasterMaterial, true, true, true);
		ResetKeywords(adaptiveTriangleBlurMaterial, true, true, true);
	}

	public void ApplyVertexWeldingIndirection(ComputeBuffer primitiveBufferToUse)
	{
		int kernelToUse = kernelWeldVertices;
		mutationOptimizerCS.SetBuffer(kernelToUse, "_PrimitiveBuffer", primitiveBufferToUse);
		mutationOptimizerCS.SetBuffer(kernelToUse, "_StructuralEdgeClosestNeighbourBuffer", structuralEdgeClosestNeighbourBuffer);
		mutationOptimizerCS.SetBuffer(kernelToUse, "_StructuralVertexWeldingBuffer", structuralVertexWeldingBuffer);
		DispatchCompute1D(mutationOptimizerCS, kernelToUse, primitiveBufferToUse.count, 256);
	}




	// ======================== MANAGEMENT ========================
	public void SetSharedComputeFrameParameters(ComputeShader computeShader)
	{
		// Set shared parameters
		computeShader.SetFloat("_IsOptim2D", targetMode == TargetMode.Image ? 1.0f : 0.0f);
		computeShader.SetInt("_CurrentOptimStep", currentOptimStep);
		computeShader.SetInt("_OutputWidth", internalOptimResolution.x);
		computeShader.SetInt("_OutputHeight", internalOptimResolution.y);
		computeShader.SetInt("_OptimizerMode", optimizer == Optimizer.RMSProp ? 0 : 1);
		computeShader.SetInt("_LossMode", lossMode == LossMode.L1 ? 0 : 1);
		computeShader.SetFloat("_OptimizerBeta1", beta1);
		computeShader.SetFloat("_OptimizerBeta2", beta2);
		computeShader.SetFloat("_MinPrimitiveWorldArea", minPrimitiveWorldArea);
		computeShader.SetInt("_FramesUnseenBeforeKill", optimStepsUnseenBeforeKill);
		computeShader.SetInt("_ViewsPerOptimStep", targetMode == TargetMode.Image ? 1 : viewsPerOptimStep);
		computeShader.SetFloat("_DoAlphaLoss", doAlphaLoss ? 1.0f : 0.0f);
		computeShader.SetFloat("_AlphaContributingCutoff", alphaContributingCutoff);

		computeShader.SetInt("_MutationsPerFrame", antitheticMutationsPerFrame);
		computeShader.SetInt("_MutationsPerFrame", antitheticMutationsPerFrame);
		computeShader.SetFloat("_OptimSuperSampling", optimResolutionFactor);
		computeShader.SetFloat("_CurrentOptimizerMipLevel", math.log2(optimResolutionFactor));
		computeShader.SetInt("_MaxFragmentsPerPixel", maxFragmentsPerPixel);
		computeShader.SetInt("_PrimitiveCount", primitiveBuffer[0].count);

		computeShader.SetVector("_ColmapTrimAABBMin", colmapTrimBounds.min);
		computeShader.SetVector("_ColmapTrimAABBMax", colmapTrimBounds.max);

		computeShader.SetInt("_BackgroundMode", (int)backgroundMode);
		computeShader.SetVector("_BackgroundColor", backgroundColor);
		computeShader.SetInt("_EnvMapResolution", envMapResolution);

		computeShader.SetFloat("_StructuralLossWeight", structuralLossWeight);
		computeShader.SetFloat("_DoStructuralLoss", doStructuralLoss == true ? 1.0f : 0.0f);
		computeShader.SetFloat("_StructuralLossDistFactor", structuralLossDistFactor);
		computeShader.SetFloat("_StructuralWeldDistFactor", structuralWeldDistFactor);
		computeShader.SetFloat("_DoPixelCountNorm", pixelCountNormalization == true ? 1.0f : 0.0f);
		computeShader.SetFloat("_DoVertexWelding", doStructuralWelding == true ? 1.0f : 0.0f);
	}

	public void RandomizeCameraView()
	{
		Bounds targetBounds = mesh3DSceneBounds;
		float distance = targetBounds.extents.magnitude;
		distance *= (randomViewZoomRange.x + UnityEngine.Random.value * (randomViewZoomRange.y - randomViewZoomRange.x));
		cameraOptim.transform.position = targetBounds.center + UnityEngine.Random.onUnitSphere * distance;
		float3 lookAtCenter = new float3(targetBounds.center) + (new float3(UnityEngine.Random.value, UnityEngine.Random.value, UnityEngine.Random.value) * 2.0f - 1.0f) * targetBounds.extents * 0.5f;
		cameraOptim.transform.LookAt(lookAtCenter, UnityEngine.Random.onUnitSphere);
		//cameraOptim.orthographicSize = (randomViewZoomRange.x + UnityEngine.Random.value * (randomViewZoomRange.y - randomViewZoomRange.x)) * targetBounds.extents.magnitude;

		// LAURENT TEST
		//cameraOptim.transform.position = targetBounds.center + UnityEngine.Random.onUnitSphere * 27.688550949f;
		//cameraOptim.transform.LookAt(Vector3.zero, UnityEngine.Random.onUnitSphere);
	}

	public void SetCameraOptimMatrices()
	{
		Vector3 cameraPos = cameraOptim.transform.position;
		Matrix4x4 cameraVP = GL.GetGPUProjectionMatrix(cameraOptim.projectionMatrix, true) * cameraOptim.worldToCameraMatrix;
		Matrix4x4 invCameraVP = cameraVP.inverse;
		mutationOptimizerCS.SetVector("_CameraWorldPos", cameraPos);
		rasterMaterial.SetVector("_CameraWorldPos", cameraPos);
		mutationOptimizerCS.SetMatrix("_CameraMatrixVP", cameraVP);
		rasterMaterial.SetMatrix("_CameraMatrixVP", cameraVP);
		mutationOptimizerCS.SetMatrix("_InvCameraMatrixVP", invCameraVP);
		rasterMaterial.SetMatrix("_InvCameraMatrixVP", invCameraVP);

		// Voxel grid mode
		Matrix4x4 invCameraV = cameraOptim.cameraToWorldMatrix;
		Matrix4x4 invCameraP = Matrix4x4.Inverse(GL.GetGPUProjectionMatrix(cameraOptim.projectionMatrix, false));
		mutationOptimizerCS.SetMatrix("_CameraMatrixInvV", invCameraV);
		mutationOptimizerCS.SetMatrix("_CameraMatrixInvP", invCameraP);
	}

	public void ResetEverything()
	{
		OnDisable();
		OnEnable();

		if (targetMode == TargetMode.COLMAP)
		{
			string folderPath = Application.dataPath.Replace("/triangle_soup_reconstruction/Assets", "") + "/Datasets/" + target3DCOLMAPFolder;
			if (File.Exists(folderPath + "/sparse/0/images.bin") == true)
				LoadAllCOLMAPDataFromBinaryFiles();
			else if (File.Exists(folderPath + "/sparse/0/images.txt") == true)
				LoadAllCOLMAPDataFromCustomBlenderTextFiles();
			else
				UnityEngine.Debug.LogError("Dataset not found/recognized");
			// Do alpha masking
			if (colmapUseMasking == true)
			{
				ApplyMaskingToCOLMAPImages();
			}
			else if(colmapInitPointPositions != null)
			{
				colmapInitPointMinDistances = new float[colmapInitPointPositions.Length];
				for (int i = 0; i < colmapInitPointMinDistances.Length; i++)
					colmapInitPointMinDistances[i] = primitiveInitSize;
				if (colmapInitPointsTrimAABBCenterSize.w > 0.0f)
					TrimColmapInitPointsWithAABB();
			}
		}

		if (targetMode == TargetMode.Model)
		{
			target3DMesh.SetActive(true);
			mesh3DSceneBounds = new Bounds();
			MeshRenderer[] allRenderers = FindObjectsOfType<MeshRenderer>();
			for (int i = 0; i < allRenderers.Length; i++)
				if (allRenderers[i].gameObject.activeInHierarchy == true)
					mesh3DSceneBounds.Encapsulate(allRenderers[i].bounds);
		}

		// Primitive buffer
		InitPrimitiveBuffer();

		// Special Env Map mode
		if (backgroundMode == BackgroundMode.EnvMap)
		{
			InitEnvMapPrimitiveBuffer(ref primitiveBuffer[1]);
		}

		cameraDisplay.GetComponent<OrbitCamera>().target = mesh3DSceneBounds;
		ResetKeywords(primitiveRendererCS, true, true, true);
		ResetKeywords(mutationOptimizerCS, true, true, true);
		InitAllOptimBuffers();
		ResetOptimizationStep(0);
		if (backgroundMode == BackgroundMode.EnvMap)
			ResetOptimizationStep(1);
		ResetKeywords(rasterMaterial, true, true, true);
		ResetKeywords(adaptiveTriangleBlurMaterial, true, true, true);

		// Set up 3D view camera
		cameraOptim.enabled = false;
		cameraOptim.targetTexture = targetFrameBuffer;
		if (targetMode == TargetMode.Image)
		{
			cameraOptim.orthographic = true;
			cameraDisplay.orthographic = true;
			cameraDisplay.transform.position = new Vector3(0.5f, 0.5f, -2.0f);
			cameraDisplay.transform.LookAt(new Vector3(0.5f, 0.5f, 0.5f));
			cameraDisplay.orthographicSize = 0.5f;
		}
		else
		{
			cameraDisplay.orthographic = false;
		}

		currentOptimStep = 0;
		currentViewPoint = 0;
		totalElapsedSeconds = 0.0f;
		systemTimer.Restart();
	}

	public void InitAllOptimBuffers()
	{
		ReleaseOptimBuffers();

		// Compute actual needed optim resolution
		Vector2 actualResolutionFloat;
		if (targetMode != TargetMode.COLMAP)
			actualResolutionFloat = targetResolution;
		else
			actualResolutionFloat = colmapViewResolution;
		actualResolutionFloat *= optimResolutionFactor;
		Vector2Int actualResolution = new Vector2Int((int)actualResolutionFloat.x, (int)actualResolutionFloat.y);

		// Display it
		internalOptimResolution = actualResolution;

		// Init rendering material
		if (optimPrimitive == PrimitiveType.TrianglesSolidUnlit || optimPrimitive == PrimitiveType.TrianglesGradientUnlit || optimPrimitive == PrimitiveType.TrianglesGaussianUnlit)
		{
			if (transparencyMode == TransparencyMode.None)
				rasterMaterial = new Material(Shader.Find("Custom/TrianglePrimitiveRaster"));
			else if (transparencyMode == TransparencyMode.SortedAlpha)
				rasterMaterial = new Material(Shader.Find("Custom/TrianglePrimitiveRasterAlphaSorted"));
			else if (transparencyMode == TransparencyMode.StochasticAlpha)
				rasterMaterial = new Material(Shader.Find("Custom/TrianglePrimitiveRasterAlphaStochastic"));
		}
		if (rasterMaterial == null) // Quick hack
		{
			rasterMaterial = new Material(Shader.Find("Custom/TrianglePrimitiveRaster"));
		}
		rasterMaterial.hideFlags = HideFlags.DontSave;
		envMapMaterial = new Material(Shader.Find("Custom/EnvMapPrimitiveRaster"));
		envMapMaterial.hideFlags = HideFlags.DontSave;

		// Init everything
		RenderTextureFormat resolveFormat = (transparencyMode == TransparencyMode.StochasticAlpha) ? RenderTextureFormat.ARGBFloat : RenderTextureFormat.ARGB32;
		//resolveFormat = RenderTextureFormat.ARGBFloat;
		if (optimRenderTargetMutatedMinus != null)
			optimRenderTargetMutatedMinus.Release();
		optimRenderTargetMutatedMinus = new RenderTexture(actualResolution.x, actualResolution.y, 32, RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear);
		optimRenderTargetMutatedMinus.enableRandomWrite = true;
		if (optimRenderTarget != null)
			optimRenderTarget.Release();
		optimRenderTarget = new RenderTexture(actualResolution.x, actualResolution.y, 32, RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear);
		optimRenderTarget.enableRandomWrite = true;
		resolvedFrameFreeView = new RenderTexture(actualResolution.x, actualResolution.y, 32, resolveFormat, RenderTextureReadWrite.Linear);
		resolvedFrameFreeView.enableRandomWrite = true;
		resolvedFrameFreeView.useMipMap = true;
		resolvedFrameFreeView.autoGenerateMips = false;
		resolvedFrameMutatedMinus = new RenderTexture(actualResolution.x, actualResolution.y, 32, resolveFormat, RenderTextureReadWrite.Linear);
		resolvedFrameMutatedMinus.enableRandomWrite = true;
		resolvedFrameMutatedMinus.useMipMap = true;
		resolvedFrameMutatedMinus.autoGenerateMips = false;
		resolvedFrameMutatedPlus = new RenderTexture(actualResolution.x, actualResolution.y, 32, resolveFormat, RenderTextureReadWrite.Linear);
		resolvedFrameMutatedPlus.enableRandomWrite = true;
		resolvedFrameMutatedPlus.useMipMap = true;
		resolvedFrameMutatedPlus.autoGenerateMips = false;
		tempResolvedFrameBuffer = new RenderTexture(actualResolution.x, actualResolution.y, 32, resolveFormat, RenderTextureReadWrite.Linear);
		tempResolvedFrameBuffer.enableRandomWrite = true;
		tempResolvedFrameBuffer.useMipMap = true;
		tempResolvedFrameBuffer.autoGenerateMips = false;
		tempResolvedFrameBuffer.filterMode = FilterMode.Bilinear;
		targetFrameBuffer = new RenderTexture(actualResolution.x, actualResolution.y, 32, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
		targetFrameBuffer.useMipMap = true;
		targetFrameBuffer.autoGenerateMips = false;
		cameraOptim.targetTexture = targetFrameBuffer;

		// Resampling buffers
		appendValidIDsBuffer = new ComputeBuffer(primitiveCount, sizeof(uint), ComputeBufferType.Append);
		argsValidIDsBuffer = new ComputeBuffer(4, sizeof(int), ComputeBufferType.IndirectArguments);
		appendInvalidIDsBuffer = new ComputeBuffer(primitiveCount, sizeof(uint), ComputeBufferType.Append);
		argsInvalidIDsBuffer = new ComputeBuffer(4, sizeof(int), ComputeBufferType.IndirectArguments);
		argsResampling = new ComputeBuffer(4, sizeof(int), ComputeBufferType.IndirectArguments);

		// Optim buffers
		int primitiveByteSize = GetPrimitiveFloatSize(optimPrimitive) * 4;
		optimStepGradientsBuffer[0] = new ComputeBuffer(primitiveCount, primitiveByteSize);
		gradientMoments1Buffer[0] = new ComputeBuffer(primitiveCount, primitiveByteSize);
		gradientMoments2Buffer[0] = new ComputeBuffer(primitiveCount, primitiveByteSize);
		primitiveBufferMutated[0] = new ComputeBuffer(primitiveCount, primitiveByteSize);
		optimStepMutationError[0] = new ComputeBuffer(primitiveCount * (trianglePerVertexError ? 3 : 1), sizeof(int) * 4);
		optimStepCounterBuffer[0] = new ComputeBuffer(primitiveCount, sizeof(int));
		primitiveKillCounters = new ComputeBuffer(primitiveCount, sizeof(int));
		ZeroInitBuffer(optimStepGradientsBuffer[0]);
		ZeroInitBuffer(gradientMoments1Buffer[0]);
		ZeroInitBuffer(gradientMoments2Buffer[0]);
		ZeroInitBuffer(primitiveBufferMutated[0]);
		ZeroInitBuffer(optimStepMutationError[0]);
		ZeroInitBuffer(optimStepCounterBuffer[0]);

		// Structural loss mode
		if (doStructuralLoss == true || doStructuralWelding == true)
		{
			ulong[] temp0 = new ulong[primitiveCount * 3];
			for (int i = 0; i < primitiveCount * 3; i++)
				temp0[i] = ulong.MaxValue;
			structuralEdgeClosestNeighbourBuffer = new ComputeBuffer(primitiveCount * 3, sizeof(ulong));
			structuralEdgeClosestNeighbourBuffer.SetData(temp0);

			int[] temp1 = new int[primitiveCount * 3];
			for (int i = 0; i < primitiveCount * 3; i++)
				temp1[i] = i;
			structuralVertexWeldingBuffer = new ComputeBuffer(primitiveCount * 3, sizeof(int));
			structuralVertexWeldingBuffer.SetData(temp1);
		}

		// Special Env Map mode
		if (backgroundMode == BackgroundMode.EnvMap)
		{
			int primitiveByteSize2 = 3 * 4;
			optimStepGradientsBuffer[1] = new ComputeBuffer(envMapResolution * envMapResolution, primitiveByteSize2);
			gradientMoments1Buffer[1] = new ComputeBuffer(envMapResolution * envMapResolution, primitiveByteSize2);
			gradientMoments2Buffer[1] = new ComputeBuffer(envMapResolution * envMapResolution, primitiveByteSize2);
			primitiveBufferMutated[1] = new ComputeBuffer(envMapResolution * envMapResolution, primitiveByteSize2);
			optimStepMutationError[1] = new ComputeBuffer(envMapResolution * envMapResolution, sizeof(int) * 3);
			optimStepCounterBuffer[1] = new ComputeBuffer(envMapResolution * envMapResolution, sizeof(int));
			ZeroInitBuffer(optimStepGradientsBuffer[1]);
			ZeroInitBuffer(gradientMoments1Buffer[1]);
			ZeroInitBuffer(gradientMoments2Buffer[1]);
			ZeroInitBuffer(primitiveBufferMutated[1]);
			ZeroInitBuffer(optimStepMutationError[1]);
			ZeroInitBuffer(optimStepCounterBuffer[1]);
		}

		// Init kill counters
		if (true)
		{
			mutationOptimizerCS.SetInt("_PrimitiveCount", primitiveBuffer[0].count);
			mutationOptimizerCS.SetInt("_FramesUnseenBeforeKill", optimStepsUnseenBeforeKill);
			mutationOptimizerCS.SetBuffer(kernelResetVisibilityCounter, "_PrimitiveKillCounters", primitiveKillCounters);
			DispatchCompute1D(mutationOptimizerCS, kernelResetVisibilityCounter, primitiveBuffer[0].count, 256);
		}

		// Transparency optim buffers
		if (transparencyMode != TransparencyMode.None)
		{
			perPixelFragmentCounterBuffer = new ComputeBuffer(actualResolution.x * actualResolution.y, sizeof(int));
			perPixelFragmentListBuffer = new ComputeBuffer(actualResolution.x * actualResolution.y, sizeof(float) * 4 * maxFragmentsPerPixel);
		}
		int sortCount = (int)(math.ceilpow2(primitiveCount));
		sortedValidPrimitiveIDBuffer = new ComputeBuffer(sortCount, sizeof(uint) * 2);
		if (transparencyMode == TransparencyMode.SortedAlpha)
		{
			// AMD FFX Sort
			sortedPrimitiveDistanceBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, primitiveCount, 4) { name = "GaussianSplatSortDistances" };
			sortedPrimitiveIDBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, primitiveCount, 4) { name = "GaussianSplatSortIndices" };
			m_Sorter = new GpuSorting(primitiveRendererCS);
			m_SorterArgs.inputKeys = sortedPrimitiveDistanceBuffer;
			m_SorterArgs.inputValues = sortedPrimitiveIDBuffer;
			m_SorterArgs.count = (uint)primitiveCount;
			if (m_Sorter.Valid)
				m_SorterArgs.resources = GpuSorting.SupportResources.Load((uint)primitiveCount);
		}

		// Render target image only once in 2D
		if (targetMode == TargetMode.Image)
		{
			Graphics.Blit(target2DImage, targetFrameBuffer);
			targetFrameBuffer.GenerateMips();
		}

		// Update watch variables
		setOptimSupersampling = optimResolutionFactor;
		setTransparencyMode = transparencyMode;
		setMaxFragmentsPerPixel = maxFragmentsPerPixel;
		setResolutionX = targetResolution.x;
		setResolutionY = targetResolution.y;
	}

	public bool CheckKeywordsValidity(ComputeShader computeShader, bool checkOptimPrimitive, bool checkTransparencyMode, bool checkSphericalHarmonicsMode)
	{
		string[] temp = computeShader.shaderKeywords;
		List<string> keywords = new List<string>(temp);
		if (keywords.Count == 0)
			return false;

		if (checkOptimPrimitive == true)
		{
			bool TRIANGLE_SOLID = keywords.Contains("TRIANGLE_SOLID");
			bool TRIANGLE_GRADIENT = keywords.Contains("TRIANGLE_GRADIENT");
			bool TRIANGLE_GAUSSIAN = keywords.Contains("TRIANGLE_GAUSSIAN");
			bool PER_VERTEX_ERROR = keywords.Contains("PER_VERTEX_ERROR");
			bool PER_TRIANGLE_ERROR = keywords.Contains("PER_TRIANGLE_ERROR");
			if (TRIANGLE_SOLID != (optimPrimitive == PrimitiveType.TrianglesSolidUnlit))
				return false;
			if (TRIANGLE_GRADIENT != (optimPrimitive == PrimitiveType.TrianglesGradientUnlit))
				return false;
			if (TRIANGLE_GAUSSIAN != (optimPrimitive == PrimitiveType.TrianglesGaussianUnlit))
				return false;
			if (PER_VERTEX_ERROR != trianglePerVertexError)
				return false;
			if (PER_TRIANGLE_ERROR != !trianglePerVertexError)
				return false;
		}

		if (checkTransparencyMode)
		{
			bool OPAQUE_RENDER = keywords.Contains("OPAQUE_RENDER");
			bool SORTED_ALPHA_RENDER = keywords.Contains("SORTED_ALPHA_RENDER");
			bool STOCHASTIC_ALPHA_RENDER = keywords.Contains("STOCHASTIC_ALPHA_RENDER");
			if (OPAQUE_RENDER != (transparencyMode == TransparencyMode.None))
				return false;
			if (SORTED_ALPHA_RENDER != (transparencyMode == TransparencyMode.SortedAlpha))
				return false;
			if (STOCHASTIC_ALPHA_RENDER != (transparencyMode == TransparencyMode.StochasticAlpha))
				return false;
		}

		if (checkSphericalHarmonicsMode == true)
		{
			bool SINGLE_COLOR = keywords.Contains("SINGLE_COLOR");
			bool SPHERICAL_HARMONICS_2 = keywords.Contains("SPHERICAL_HARMONICS_2");
			bool SPHERICAL_HARMONICS_3 = keywords.Contains("SPHERICAL_HARMONICS_3");
			bool SPHERICAL_HARMONICS_4 = keywords.Contains("SPHERICAL_HARMONICS_4");
			if (SINGLE_COLOR != (sphericalHarmonicsMode == SphericalHarmonicsMode.None))
				return false;
			if (SPHERICAL_HARMONICS_2 != (sphericalHarmonicsMode == SphericalHarmonicsMode.TwoBands))
				return false;
			if (SPHERICAL_HARMONICS_3 != (sphericalHarmonicsMode == SphericalHarmonicsMode.ThreeBands))
				return false;
			if (SPHERICAL_HARMONICS_4 != (sphericalHarmonicsMode == SphericalHarmonicsMode.FourBands))
				return false;
		}

		return true;
	}

	public void ResetKeywords(ComputeShader computeShader, bool doOptimPrimitive, bool doTransparencyMode, bool checkSphericalHarmonicsMode)
	{
		if (doOptimPrimitive == true)
		{
			computeShader.DisableKeyword("TRIANGLE_SOLID");
			computeShader.DisableKeyword("TRIANGLE_GRADIENT");
			computeShader.DisableKeyword("TRIANGLE_GAUSSIAN");
			computeShader.DisableKeyword("PER_VERTEX_ERROR");
			computeShader.DisableKeyword("PER_TRIANGLE_ERROR");
				computeShader.EnableKeyword("ALTERNATE_POSITIONS");
			if (optimPrimitive == PrimitiveType.TrianglesSolidUnlit)
				computeShader.EnableKeyword("TRIANGLE_SOLID");
			else if (optimPrimitive == PrimitiveType.TrianglesGradientUnlit)
				computeShader.EnableKeyword("TRIANGLE_GRADIENT");
			else if (optimPrimitive == PrimitiveType.TrianglesGaussianUnlit)
				computeShader.EnableKeyword("TRIANGLE_GAUSSIAN");
			if (trianglePerVertexError == false)
				computeShader.EnableKeyword("PER_TRIANGLE_ERROR");
			if (trianglePerVertexError == true)
				computeShader.EnableKeyword("PER_VERTEX_ERROR");
		}

		if (doTransparencyMode == true)
		{
			computeShader.DisableKeyword("OPAQUE_RENDER");
			computeShader.DisableKeyword("SORTED_ALPHA_RENDER");
			computeShader.DisableKeyword("STOCHASTIC_ALPHA_RENDER");
			if (transparencyMode == TransparencyMode.None)
				computeShader.EnableKeyword("OPAQUE_RENDER");
			else if (transparencyMode == TransparencyMode.SortedAlpha)
				computeShader.EnableKeyword("SORTED_ALPHA_RENDER");
			else
				computeShader.EnableKeyword("STOCHASTIC_ALPHA_RENDER");
		}

		if (checkSphericalHarmonicsMode == true)
		{
			computeShader.DisableKeyword("SINGLE_COLOR");
			computeShader.DisableKeyword("SPHERICAL_HARMONICS_2");
			computeShader.DisableKeyword("SPHERICAL_HARMONICS_3");
			computeShader.DisableKeyword("SPHERICAL_HARMONICS_4");
			if (sphericalHarmonicsMode == SphericalHarmonicsMode.None)
				computeShader.EnableKeyword("SINGLE_COLOR");
			else if (sphericalHarmonicsMode == SphericalHarmonicsMode.TwoBands)
				computeShader.EnableKeyword("SPHERICAL_HARMONICS_2");
			else if (sphericalHarmonicsMode == SphericalHarmonicsMode.ThreeBands)
				computeShader.EnableKeyword("SPHERICAL_HARMONICS_3");
			else
				computeShader.EnableKeyword("SPHERICAL_HARMONICS_4");
		}
	}

	public void ResetKeywords(Material material, bool doOptimPrimitive, bool doTransparencyMode, bool checkSphericalHarmonicsMode)
	{
		if (material == null)
			return;

		if (doOptimPrimitive == true)
		{
			material.DisableKeyword("TRIANGLE_SOLID");
			material.DisableKeyword("TRIANGLE_GRADIENT");
			material.DisableKeyword("TRIANGLE_GAUSSIAN");
			material.DisableKeyword("PER_VERTEX_ERROR");
			material.DisableKeyword("PER_TRIANGLE_ERROR");
				material.EnableKeyword("ALTERNATE_POSITIONS");
			if (optimPrimitive == PrimitiveType.TrianglesSolidUnlit)
				material.EnableKeyword("TRIANGLE_SOLID");
			else if (optimPrimitive == PrimitiveType.TrianglesGradientUnlit)
				material.EnableKeyword("TRIANGLE_GRADIENT");
			else if (optimPrimitive == PrimitiveType.TrianglesGaussianUnlit)
				material.EnableKeyword("TRIANGLE_GAUSSIAN");
			if (trianglePerVertexError == false)
				material.EnableKeyword("PER_TRIANGLE_ERROR");
			if (trianglePerVertexError == true)
				material.EnableKeyword("PER_VERTEX_ERROR");
		}

		if (doTransparencyMode == true)
		{
			material.DisableKeyword("OPAQUE_RENDER");
			material.DisableKeyword("SORTED_ALPHA_RENDER");
			material.DisableKeyword("STOCHASTIC_ALPHA_RENDER");
			if (transparencyMode == TransparencyMode.None)
				material.EnableKeyword("OPAQUE_RENDER");
			else if (transparencyMode == TransparencyMode.SortedAlpha)
				material.EnableKeyword("SORTED_ALPHA_RENDER");
			else
				material.EnableKeyword("STOCHASTIC_ALPHA_RENDER");
		}

		if (checkSphericalHarmonicsMode == true)
		{
			material.DisableKeyword("SINGLE_COLOR");
			material.DisableKeyword("SPHERICAL_HARMONICS_2");
			material.DisableKeyword("SPHERICAL_HARMONICS_3");
			material.DisableKeyword("SPHERICAL_HARMONICS_4");
			if (sphericalHarmonicsMode == SphericalHarmonicsMode.None)
				material.EnableKeyword("SINGLE_COLOR");
			else if (sphericalHarmonicsMode == SphericalHarmonicsMode.TwoBands)
				material.EnableKeyword("SPHERICAL_HARMONICS_2");
			else if (sphericalHarmonicsMode == SphericalHarmonicsMode.ThreeBands)
				material.EnableKeyword("SPHERICAL_HARMONICS_3");
			else
				material.EnableKeyword("SPHERICAL_HARMONICS_4");
		}
	}

	public bool ChecksResetEverythingConditions()
	{
		if (primitiveBuffer[0] == null
			|| setPrimitiveCount != primitiveCount
			|| setTargetMode != targetMode
			|| setOptimPrimitive != optimPrimitive
			|| rasterMaterial == null)
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	public void ResetWatchVariables()
	{
		setPrimitiveCount = primitiveCount;
		setTargetMode = targetMode;
		setOptimPrimitive = optimPrimitive;
		setPrimitiveResampling = doPrimitiveResampling;
	}

	private static void DispatchCompute1D(ComputeShader compute, int kernel, int threadCount, int groupSizeX)
	{
		int threadGroupCount = (int)math.ceil(threadCount / (float)groupSizeX);
		int dispatchCount = (int)math.ceil(threadGroupCount / 65536.0f);
		int offset = 0;
		for (int i = 0; i < dispatchCount; i++)
		{
			int currentGroupCount = (int)math.min(threadGroupCount - offset, 65535.0f);
			compute.SetInt("_Dispatch1DOffset", offset);
			compute.Dispatch(kernel, currentGroupCount, 1, 1);
			offset += currentGroupCount;
		}
	}

	public Mesh GenerateQuadMesh(int resolution)
	{
		float size = 1.0f;
		var vertices = new List<Vector3>();
		var normals = new List<Vector3>();
		var uvs = new List<Vector2>();
		for (int y = 0; y < resolution; y++)
		{
			for (int x = 0; x < resolution; x++)
			{
				float tx = (float)x / (float)(resolution - 1);
				float ty = (float)y / (float)(resolution - 1);
				vertices.Add(new Vector3(-0.5f + tx, 0, -0.5f + ty) * size);
				normals.Add(Vector3.up);
				uvs.Add(new Vector2(tx, ty));
			}
		}

		var indices = new List<int>();
		for (int y = 0; y < (resolution - 1); y++)
		{
			for (int x = 0; x < (resolution - 1); x++)
			{
				int quad = y * resolution + x;

				indices.Add(quad);
				indices.Add(quad + resolution);
				indices.Add(quad + resolution + 1);

				indices.Add(quad);
				indices.Add(quad + resolution + 1);
				indices.Add(quad + 1);
			}
		}

		Mesh mesh = new Mesh();
		mesh.indexFormat = IndexFormat.UInt32;
		mesh.vertices = vertices.ToArray();
		mesh.triangles = indices.ToArray();
		mesh.uv = uvs.ToArray();
		mesh.normals = normals.ToArray();
		mesh.RecalculateBounds();
		mesh.RecalculateTangents();
		return mesh;
	}

	public Mesh GenerateCubeMesh(int resolution)
	{
		float size = 1.0f;
		var vertices = new List<Vector3>();
		var normals = new List<Vector3>();
		var uvs = new List<Vector2>();
		var indices = new List<int>();
		for (int i = 0; i < 6; i++)
		{
			int currentOffset = vertices.Count;

			for (int y = 0; y < resolution; y++)
			{
				for (int x = 0; x < resolution; x++)
				{
					float tx = (float)x / (float)(resolution - 1);
					float ty = (float)y / (float)(resolution - 1);

					if (i == 0)
						vertices.Add(new Vector3(-0.5f + tx, -0.5f, -0.5f + ty) * size);
					else if (i == 1)
						vertices.Add(new Vector3(-0.5f + tx, 0.5f, -0.5f + ty) * size);
					else if (i == 2)
						vertices.Add(new Vector3(-0.5f, -0.5f + ty, -0.5f + tx) * size);
					else if (i == 3)
						vertices.Add(new Vector3(0.5f, -0.5f + ty, -0.5f + tx) * size);
					else if (i == 4)
						vertices.Add(new Vector3(-0.5f + ty, -0.5f + tx, -0.5f) * size);
					else
						vertices.Add(new Vector3(-0.5f + ty, -0.5f + tx, 0.5f) * size);

					if (i == 0)
						normals.Add(Vector3.down);
					else if (i == 1)
						normals.Add(Vector3.up);
					else if (i == 2)
						normals.Add(Vector3.left);
					else if (i == 3)
						normals.Add(Vector3.right);
					else if (i == 4)
						normals.Add(Vector3.back);
					else
						normals.Add(Vector3.forward);

					uvs.Add(new Vector2(tx, ty));
				}
			}

			for (int y = 0; y < (resolution - 1); y++)
			{
				for (int x = 0; x < (resolution - 1); x++)
				{
					int quad = y * resolution + x;

					if (i % 2 == 0)
					{
						indices.Add(currentOffset + quad);
						indices.Add(currentOffset + quad + resolution + 1);
						indices.Add(currentOffset + quad + resolution);
					}
					else
					{
						indices.Add(currentOffset + quad);
						indices.Add(currentOffset + quad + resolution);
						indices.Add(currentOffset + quad + resolution + 1);
					}

					if (i % 2 == 0)
					{
						indices.Add(currentOffset + quad);
						indices.Add(currentOffset + quad + 1);
						indices.Add(currentOffset + quad + resolution + 1);
					}
					else
					{
						indices.Add(currentOffset + quad);
						indices.Add(currentOffset + quad + resolution + 1);
						indices.Add(currentOffset + quad + 1);
					}
				}
			}
		}

		Mesh mesh = new Mesh();
		mesh.indexFormat = IndexFormat.UInt32;
		mesh.vertices = vertices.ToArray();
		mesh.triangles = indices.ToArray();
		mesh.uv = uvs.ToArray();
		mesh.normals = normals.ToArray();
		mesh.RecalculateBounds();
		mesh.RecalculateTangents();
		return mesh;
	}

	public void ZeroInitBuffer(ComputeBuffer buffer)
	{
		byte[] temp = new byte[buffer.count * buffer.stride];
		buffer.SetData(temp);
	}




	// ======================== IMAGE DATASETS ========================
	public void LoadAllCOLMAPDataFromBinaryFiles()
	{
		// Load images extrinsics
		string folderPath = Application.dataPath.Replace("/triangle_soup_reconstruction/Assets", "") + "/Datasets/" + target3DCOLMAPFolder;
		BinaryReader reader = new BinaryReader(File.Open(folderPath + "/sparse/0/images.bin", FileMode.Open, FileAccess.Read, FileShare.Read));
		colmapImageCount = (int)reader.ReadUInt64();
		colmapViewsTarget = new Texture2D[colmapImageCount];
		colmapViewsPos = new Vector3[colmapImageCount];
		colmapViewsRot = new Quaternion[colmapImageCount];
		for (int i = 0; i < colmapImageCount; i++)
		{
			// Read everything for this image
			int imageID = reader.ReadInt32();
			float qvecW = (float)reader.ReadDouble();
			Quaternion qvec = new Quaternion((float)reader.ReadDouble(), (float)reader.ReadDouble(), (float)reader.ReadDouble(), qvecW);
			Vector3 tvec = new Vector3((float)reader.ReadDouble(), (float)reader.ReadDouble(), (float)reader.ReadDouble());
			int cameraID = reader.ReadInt32();
			string imageName = "";
			while (imageName.Length == 0 || imageName[imageName.Length - 1] != '\0')
				imageName += reader.ReadChar();
			imageName = imageName.Remove(imageName.Length - 1);
			int numPoints2D = (int)reader.ReadUInt64();
			reader.ReadBytes(24 * numPoints2D);

			// Rescale
			tvec *= colmapRescaler;

			// Transform into our data
			colmapViewsTarget[i] = new Texture2D(2, 2);
			string[] imageNameSplit = imageName.Split('.');
			if (File.Exists(folderPath + "/images/" + imageNameSplit[0] + ".png") == true)
				imageName = imageNameSplit[0] + ".png";
			else if (File.Exists(folderPath + "/images/" + imageNameSplit[0] + ".jpg") == true)
				imageName = imageNameSplit[0] + ".jpg";
			else if (File.Exists(folderPath + "/images/" + imageNameSplit[0] + ".jpeg") == true)
				imageName = imageNameSplit[0] + ".jpeg";
			byte[] texData = File.ReadAllBytes(folderPath + "/images/" + imageName);
			colmapViewsTarget[i].LoadImage(texData);
			colmapViewsTarget[i].Apply(true);
			colmapViewsTarget[i].name = imageNameSplit[0];
			tvec.y = -tvec.y;
			qvec.y = -qvec.y;
			qvec.w = -qvec.w;
			Matrix4x4 world2Camera = Matrix4x4.TRS(tvec, Quaternion.identity, Vector3.one) * Matrix4x4.TRS(Vector3.zero, qvec, Vector3.one);
			colmapViewsPos[i] = world2Camera.inverse.GetPosition();
			colmapViewsRot[i] = world2Camera.inverse.rotation;
		}

		// Load camera intrisics
		reader = new BinaryReader(File.Open(folderPath + "/sparse/0/cameras.bin", FileMode.Open, FileAccess.Read, FileShare.Read));
		uint cameraCount = (uint)reader.ReadUInt64();
		for (int i = 0; i < cameraCount; i++) // There should be only one camera intrisics to load
		{
			// Read everything for this camera
			int cameraID = reader.ReadInt32();
			int modelID = reader.ReadInt32();
			int width = (int)reader.ReadUInt64();
			int height = (int)reader.ReadUInt64();
			float focalLengthX = (float)reader.ReadDouble();
			float focalLengthY = (float)reader.ReadDouble();
			float opticalCenterX = (float)reader.ReadDouble();
			float opticalCenterY = (float)reader.ReadDouble();

			// Setup our optim camera
			//colmapViewResolution = new Vector2Int(width, height);
			colmapViewResolution = new Vector2Int(colmapViewsTarget[0].width, colmapViewsTarget[0].height);
			float fovVertical = math.atan((height / 2.0f) / (float)focalLengthY) * 2.0f * Mathf.Rad2Deg;
			cameraOptim.orthographic = false;
			cameraOptim.fieldOfView = fovVertical;
			if (target3DCOLMAPFolder == "pinecone" || target3DCOLMAPFolder == "vasedeck") // !!!!!! PINECONE VASEDECK WORKAROUND !!!!!!
				cameraOptim.fieldOfView = 48.0f;
		}

		// Read COLMAP reconstructed points
		if (initPrimitivesOnMeshSurface == true && File.Exists(folderPath + "/sparse/0/points3D.bin") == true)
		{
			reader = new BinaryReader(File.Open(folderPath + "/sparse/0/points3D.bin", FileMode.Open, FileAccess.Read, FileShare.Read));
			int pointCount = (int)reader.ReadUInt64();
			colmapInitPointPositions = new float3[pointCount];
			colmapInitPointBounds = new Bounds();
			for (int i = 0; i < pointCount; i++)
			{
				// Read everything for this point
				uint pointID = (uint)reader.ReadUInt64();
				Vector3 pos = new Vector3((float)reader.ReadDouble(), (float)reader.ReadDouble(), (float)reader.ReadDouble());
				Color32 col = new Color32(reader.ReadByte(), reader.ReadByte(), reader.ReadByte(), 255);
				float error = (float)reader.ReadDouble();
				int trackLength = (int)reader.ReadUInt64();
				reader.ReadBytes(8 * trackLength);

				// Rescale
				pos *= colmapRescaler;

				// Store into our data
				pos.y = -pos.y;
				colmapInitPointPositions[i] = pos;
				colmapInitPointBounds.Encapsulate(pos);
			}
		}
		else
		{
			initPrimitivesOnMeshSurface = false;
			colmapInitPointBounds = new Bounds(Vector3.zero, Vector3.one);
		}

		// Camera position bounding box
		if (colmapUseCameraBounds == true) // Always use camera bounds for volume
		{
			colmapInitPointBounds = new Bounds(Vector3.zero, Vector3.zero);
			for (int i = 0; i < colmapViewsPos.Length; i++)
				colmapInitPointBounds.Encapsulate(colmapViewsPos[i]);
		}
	}

	public void LoadAllCOLMAPDataFromCustomBlenderTextFiles()
	{
		// Load images extrinsics
		string folderPath = Application.dataPath.Replace("/triangle_soup_reconstruction/Assets", "") + "/Datasets/" + target3DCOLMAPFolder;
		TextReader reader = File.OpenText(folderPath + "/sparse/0/images.txt");
		colmapImageCount = File.ReadAllLines(folderPath + "/sparse/0/images.txt").Length / 2;
		colmapViewsTarget = new Texture2D[colmapImageCount];
		colmapViewsPos = new Vector3[colmapImageCount];
		colmapViewsRot = new Quaternion[colmapImageCount];
		for (int i = 0; i < colmapImageCount; i++)
		{
			// Read everything for this image
			string[] line = reader.ReadLine().Split(" ");
			float qvecW = (float)double.Parse(line[1]);
			Quaternion qvec = new Quaternion((float)double.Parse(line[2]), (float)double.Parse(line[3]), (float)double.Parse(line[4]), qvecW);
			Vector3 tvec = new Vector3((float)double.Parse(line[5]), (float)double.Parse(line[6]), (float)double.Parse(line[7]));
			string imageName = line[9];
			reader.ReadLine();

			// Rescale
			tvec *= colmapRescaler;

			// Transform into our data
			byte[] texData = File.ReadAllBytes(folderPath + "/images/" + imageName);
			colmapViewsTarget[i] = new Texture2D(2, 2);
			colmapViewsTarget[i].LoadImage(texData);
			colmapViewsTarget[i].Apply(true);

			//colmapViewsTarget[i] = (Texture2D)Resources.Load(imageName.Split('.')[0], typeof(Texture2D)); // TEMP HACK FIRE
			//colmapViewsTarget[i].name = imageName.Split('.')[0];

			//tvec.y = -tvec.y;
			//qvec.y = -qvec.y;
			//qvec.w = -qvec.w;
			Matrix4x4 world2Camera = Matrix4x4.TRS(tvec, Quaternion.identity, Vector3.one) * Matrix4x4.TRS(Vector3.zero, qvec * Quaternion.Euler(90, 0, 0), Vector3.one);
			colmapViewsPos[i] = world2Camera.inverse.GetPosition();
			colmapViewsRot[i] = world2Camera.inverse.rotation;
			colmapViewsRot[i] *= Quaternion.Euler(0, 180, 0);
		}

		// Load camera intrisics
		reader = File.OpenText(folderPath + "/sparse/0/cameras.txt");
		for (int i = 0; i < 1; i++) // There should be only one camera intrisics to load
		{
			// Read everything for this camera
			string[] line = reader.ReadLine().Split(" ");
			int width = int.Parse(line[2]);
			int height = int.Parse(line[3]);
			float focalLengthX = (float)double.Parse(line[4]);
			float focalLengthY = (float)double.Parse(line[5]);
			float opticalCenterX = (float)double.Parse(line[6]);
			float opticalCenterY = (float)double.Parse(line[7]);

			// Setup our optim camera
			//colmapViewResolution = new Vector2Int(width, height);
			colmapViewResolution = new Vector2Int(colmapViewsTarget[0].width, colmapViewsTarget[0].height);
			float fovVertical = math.atan((height / 2.0f) / (float)focalLengthY) * 2.0f * Mathf.Rad2Deg;
			cameraOptim.orthographic = false;
			cameraOptim.fieldOfView = fovVertical;
		}

		// Read custom blender point cloud
		if (File.Exists(folderPath + "/sparse/0/points3D.txt") == true)
		{
			reader = File.OpenText(folderPath + "/sparse/0/points3D.txt");
			int pointCount = int.Parse(reader.ReadLine());
			colmapInitPointPositions = new float3[pointCount];
			colmapInitPointBounds = new Bounds();
			for (int i = 0; i < pointCount; i++)
			{
				string[] line = reader.ReadLine().Split(" ");
				Vector3 pos = new Vector3((float)double.Parse(line[0]), (float)double.Parse(line[1]), (float)double.Parse(line[2]));

				// Rescale
				pos *= colmapRescaler;

				// Store into our data
				//pos.y = -pos.y;
				pos = Matrix4x4.TRS(Vector3.zero, Quaternion.Euler(-90, 0, 0), Vector3.one) * pos;
				colmapInitPointPositions[i] = pos;
				colmapInitPointBounds.Encapsulate(pos);
			}
		}
		else
		{
			initPrimitivesOnMeshSurface = false;
			colmapInitPointBounds = new Bounds(Vector3.zero, Vector3.one);
		}

		// Camera position bounding box
		if (colmapUseCameraBounds == true)
		{
			colmapInitPointBounds = new Bounds(Vector3.zero, Vector3.zero);
			for (int i = 0; i < colmapViewsPos.Length; i++)
				colmapInitPointBounds.Encapsulate(colmapViewsPos[i]);
		}
	}

	public void LoadAllCOLMAPDataFromCustomBlenderTextFilesWriteLaurent()
	{
		// Load images extrinsics
		string folderPath = Application.dataPath.Replace("/triangle_soup_reconstruction/Assets", "") + "/Datasets/" + target3DCOLMAPFolder;
		TextReader reader = File.OpenText(folderPath + "/sparse/0/images.txt");
		StreamWriter writer = new StreamWriter(folderPath + "/sparse/0/images_custom.txt");
		colmapImageCount = File.ReadAllLines(folderPath + "/sparse/0/images.txt").Length / 2;
		colmapViewsTarget = new Texture2D[colmapImageCount];
		colmapViewsPos = new Vector3[colmapImageCount];
		colmapViewsRot = new Quaternion[colmapImageCount];
		for (int i = 0; i < colmapImageCount; i++)
		{
			// Read everything for this image
			string[] line = reader.ReadLine().Split(" ");
			float qvecW = (float)double.Parse(line[1]);
			Quaternion qvec = new Quaternion((float)double.Parse(line[2]), (float)double.Parse(line[3]), (float)double.Parse(line[4]), qvecW);
			Vector3 tvec = new Vector3((float)double.Parse(line[5]), (float)double.Parse(line[6]), (float)double.Parse(line[7]));
			string imageName = line[9];
			reader.ReadLine();

			// Rescale
			tvec *= colmapRescaler;

			// Transform into our data
			byte[] texData = File.ReadAllBytes(folderPath + "/images/" + imageName);
			colmapViewsTarget[i] = new Texture2D(2, 2);
			colmapViewsTarget[i].LoadImage(texData);
			colmapViewsTarget[i].Apply(true);
			colmapViewsTarget[i].name = imageName.Split('.')[0];
			//tvec.y = -tvec.y;
			//qvec.y = -qvec.y;
			//qvec.w = -qvec.w;
			Matrix4x4 world2Camera = Matrix4x4.TRS(tvec, Quaternion.identity, Vector3.one) * Matrix4x4.TRS(Vector3.zero, qvec * Quaternion.Euler(90, 0, 0), Vector3.one);
			colmapViewsPos[i] = world2Camera.inverse.GetPosition();
			colmapViewsRot[i] = world2Camera.inverse.rotation;
			colmapViewsRot[i] *= Quaternion.Euler(0, 180, 0);

			// Write for Laurent
			Matrix4x4 world2CameraWrite = Matrix4x4.TRS(tvec, Quaternion.identity, Vector3.one) * Matrix4x4.TRS(Vector3.zero, qvec * Quaternion.Euler(0, 0, 0), Vector3.one);
			Vector3 pos = world2Camera.inverse.GetPosition();
			writer.WriteLine((i + 1).ToString());
			writer.WriteLine(imageName);
			writer.WriteLine(world2CameraWrite.ToString());
		}
		writer.Close();

		// Load camera intrisics
		reader = File.OpenText(folderPath + "/sparse/0/cameras.txt");
		StreamWriter writer2 = new StreamWriter(folderPath + "/sparse/0/cameras_custom.txt");
		for (int i = 0; i < 1; i++) // There should be only one camera intrisics to load
		{
			// Read everything for this camera
			string[] line = reader.ReadLine().Split(" ");
			int width = int.Parse(line[2]);
			int height = int.Parse(line[3]);
			float focalLengthX = (float)double.Parse(line[4]);
			float focalLengthY = (float)double.Parse(line[5]);
			float opticalCenterX = (float)double.Parse(line[6]);
			float opticalCenterY = (float)double.Parse(line[7]);

			// Setup our optim camera
			//colmapViewResolution = new Vector2Int(width, height);
			colmapViewResolution = new Vector2Int(colmapViewsTarget[0].width, colmapViewsTarget[0].height);
			float fovVertical = math.atan((height / 2.0f) / (float)focalLengthY) * 2.0f * Mathf.Rad2Deg;
			cameraOptim.orthographic = false;
			cameraOptim.fieldOfView = fovVertical;

			// Write for Laurent
			writer2.WriteLine(width.ToString(System.Globalization.CultureInfo.InvariantCulture));
			writer2.WriteLine(height.ToString(System.Globalization.CultureInfo.InvariantCulture));
			writer2.WriteLine(focalLengthX.ToString(System.Globalization.CultureInfo.InvariantCulture));
			writer2.WriteLine(focalLengthY.ToString(System.Globalization.CultureInfo.InvariantCulture));
			writer2.WriteLine(opticalCenterX.ToString(System.Globalization.CultureInfo.InvariantCulture));
			writer2.WriteLine(opticalCenterY.ToString(System.Globalization.CultureInfo.InvariantCulture));
		}
		writer2.Close();

		// Read custom blender point cloud
		if (File.Exists(folderPath + "/sparse/0/points3D.txt") == true)
		{
			reader = File.OpenText(folderPath + "/sparse/0/points3D.txt");
			int pointCount = int.Parse(reader.ReadLine());
			colmapInitPointPositions = new float3[pointCount];
			colmapInitPointBounds = new Bounds();
			for (int i = 0; i < pointCount; i++)
			{
				string[] line = reader.ReadLine().Split(" ");
				Vector3 pos = new Vector3((float)double.Parse(line[0]), (float)double.Parse(line[1]), (float)double.Parse(line[2]));

				// Rescale
				pos *= colmapRescaler;

				// Store into our data
				//pos.y = -pos.y;
				pos = Matrix4x4.TRS(Vector3.zero, Quaternion.Euler(-90, 0, 0), Vector3.one) * pos;
				colmapInitPointPositions[i] = pos;
				colmapInitPointBounds.Encapsulate(pos);
			}
		}
		else
		{
			initPrimitivesOnMeshSurface = false;
			colmapInitPointBounds = new Bounds(Vector3.zero, Vector3.one);
		}

		// Camera position bounding box
		if (colmapUseCameraBounds == true)
		{
			colmapInitPointBounds = new Bounds(Vector3.zero, Vector3.zero);
			for (int i = 0; i < colmapViewsPos.Length; i++)
				colmapInitPointBounds.Encapsulate(colmapViewsPos[i]);
		}
	}

	public void ApplyMaskingToCOLMAPImages()
	{
		Material maskMaterial = new Material(Shader.Find("Custom/Masking"));
		string folderPath = Application.dataPath.Replace("/triangle_soup_reconstruction/Assets", "") + "/Datasets/" + target3DCOLMAPFolder;
		for (int i = 0; i < colmapImageCount; i++)
		{
			Texture2D mask = new Texture2D(2, 2);
			byte[] texData = File.ReadAllBytes(folderPath + "/masks/" + colmapViewsTarget[i].name + ".png");
			mask.LoadImage(texData);
			mask.Apply();

			// Create a new RenderTexture with the desired dimensions
			RenderTexture rt = new RenderTexture(mask.width, mask.height, 0, colmapViewsTarget[i].format == TextureFormat.RGBA32 ? RenderTextureFormat.ARGB32 : RenderTextureFormat.ARGBFloat);
			rt.filterMode = FilterMode.Bilinear;

			// Save the current active RenderTexture
			RenderTexture previous = RenderTexture.active;

			// Set the new RenderTexture as active and render the original texture onto it
			RenderTexture.active = rt;
			maskMaterial.SetTexture("_MainTex", colmapViewsTarget[i]);
			maskMaterial.SetTexture("_AlphaTex", mask);
			Graphics.Blit(null, rt, maskMaterial);

			// Create a new Texture2D and read the pixels from the RenderTexture
			colmapViewsTarget[i].ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
			colmapViewsTarget[i].Apply(true);

			// Restore the previous RenderTexture
			RenderTexture.active = previous;
			rt.Release();
		}
	}

	public void TrimColmapInitPointsWithAABB()
	{
		List<float3> prunedPositions = new List<float3>();
		List<float> prunedDistances = new List<float>();
		for (int i = 0; i < colmapInitPointPositions.Length; i++)
		{
			if (colmapTrimBounds.Contains(colmapInitPointPositions[i]) == true)
			{
				prunedPositions.Add(colmapInitPointPositions[i]);
				if (colmapInitPointMinDistances != null && colmapInitPointMinDistances.Length > 0)
					prunedDistances.Add(colmapInitPointMinDistances[i]);
			}
		}
		colmapInitPointPositions = prunedPositions.ToArray();
		colmapInitPointMinDistances = prunedDistances.ToArray();
	}




	// ======================= PRIMITIVE INIT =======================
	public int GetPrimitiveFloatSize(PrimitiveType type)
	{
		int positionSize = 9;

		int alphaSize = 0;
		if (transparencyMode != TransparencyMode.None)
			alphaSize = 1;

		int colorSize = 0;
		if (sphericalHarmonicsMode == SphericalHarmonicsMode.None)
			colorSize = 3;
		else if (sphericalHarmonicsMode == SphericalHarmonicsMode.TwoBands)
			colorSize = 12;
		else if (sphericalHarmonicsMode == SphericalHarmonicsMode.ThreeBands)
			colorSize = 27;
		else if (sphericalHarmonicsMode == SphericalHarmonicsMode.FourBands)
			colorSize = 48;

		if (type == PrimitiveType.TrianglesSolidUnlit)
			return positionSize + colorSize + alphaSize;
		else if (type == PrimitiveType.TrianglesGradientUnlit)
			return positionSize + colorSize * 3 + alphaSize * 3;
		else if (type == PrimitiveType.TrianglesGaussianUnlit)
			return positionSize + colorSize + alphaSize;
		else
			return 0;
	}

	public void InitPrimitiveBuffer()
	{
		// Random seed
		if (primitiveInitSeed < 0)
			primitiveInitSeed = (int)(UnityEngine.Random.value * int.MaxValue);
		UnityEngine.Random.InitState(primitiveInitSeed);

		// Init random positions
		float3[] positions = new float3[primitiveCount];
		float[] initSizes = new float[0];
		if (optimPrimitive == PrimitiveType.TrianglesSolidUnlit || optimPrimitive == PrimitiveType.TrianglesGradientUnlit || optimPrimitive == PrimitiveType.TrianglesGaussianUnlit)
		{
			if (targetMode == TargetMode.Model && initPrimitivesOnMeshSurface == true)
				InitPositionsOnMeshSurface(positions);
			else if (targetMode == TargetMode.COLMAP && initPrimitivesOnMeshSurface == true)
				InitPositionsOnCOLMAPPoints(ref positions, ref initSizes);
			else
				InitPositionsInsideBounds(positions);
		}

		// Init primitive buffer
		InitTrianglePrimitiveBuffer(ref primitiveBuffer[0], positions, initSizes);
	}

	public void InitPositionsInsideBounds(float3[] positions)
	{
		// Init random primitives within target bounds
		Bounds targetBounds;
		if (targetMode == TargetMode.Image)
			targetBounds = new Bounds(Vector3.one * 0.5f, Vector3.one);
		else if (targetMode == TargetMode.Model)
			targetBounds = mesh3DSceneBounds;
		else
			targetBounds = colmapInitPointBounds;

		for (int i = 0; i < primitiveCount; i++)
		{
			float3 randPos = new float3(UnityEngine.Random.value, UnityEngine.Random.value, UnityEngine.Random.value) * 2.0f - 1.0f;
			randPos = new float3(targetBounds.center) + randPos * new float3(targetBounds.extents);
			positions[i] = randPos;
		}
	}

	public void InitPositionsOnMeshSurface(float3[] positions)
	{
		// Init random triangles on target mesh surface
		Vector3[] targetVertices = target3DMesh.GetComponent<MeshFilter>().sharedMesh.vertices;
		int[] triangles = target3DMesh.GetComponent<MeshFilter>().sharedMesh.triangles;
		int targetTriangleCount = triangles.Length / 3;

		for (int i = 0; i < primitiveCount; i++)
		{
			// Select random triangle
			int triangleID = (int)math.min(targetTriangleCount - 1, UnityEngine.Random.value * targetTriangleCount);
			Vector3 vertexA = target3DMesh.transform.TransformPoint(targetVertices[triangles[triangleID * 3 + 0]]);
			Vector3 vertexB = target3DMesh.transform.TransformPoint(targetVertices[triangles[triangleID * 3 + 1]]);
			Vector3 vertexC = target3DMesh.transform.TransformPoint(targetVertices[triangles[triangleID * 3 + 2]]);

			// Random position in triangle
			float2 randTriangle = new float2(UnityEngine.Random.value, UnityEngine.Random.value);
			if (randTriangle.x + randTriangle.y >= 1)
				randTriangle = 1.0f - randTriangle;
			float3 randPos = vertexA + randTriangle.x * (vertexB - vertexA) + randTriangle.y * (vertexC - vertexA);
			positions[i] = randPos;
		}
	}

	public void InitPositionsOnCOLMAPPoints(ref float3[] positions, ref float[] initSizes)
	{
		initSizes = new float[primitiveCount];
		int pointCount = colmapInitPointPositions.Length;
		if (colmapOnePrimitivePerInitPoint == true)
		{
			primitiveCount = pointCount;
			positions = new float3[primitiveCount];
			initSizes = new float[primitiveCount];
		}
		for (int i = 0; i < primitiveCount; i++)
		{
			// Select random point
			int pointID = (int)math.min(pointCount - 1, UnityEngine.Random.value * pointCount);
			if (colmapOnePrimitivePerInitPoint == true)
				pointID = i % pointCount;
			float3 randPos = new float3(colmapInitPointPositions[pointID]);
			if (colmapOnePrimitivePerInitPoint == false)
				randPos += new float3(UnityEngine.Random.insideUnitSphere) * colmapInitPointMinDistances[pointID] * 0.5f;
			positions[i] = randPos;
			initSizes[i] = colmapInitPointMinDistances[pointID];
		}
	}

	public void InitTrianglePrimitiveBuffer(ref ComputeBuffer primitiveBufferToInit, float3[] positions, float[] initSizes)
	{
		// Init data on CPU
		int primitiveFloatSize = GetPrimitiveFloatSize(optimPrimitive);
		float[] randData = new float[primitiveCount * primitiveFloatSize];

		for (int i = 0; i < primitiveCount; i++)
		{
			int offset = 0;

			// Position
			float initSizeToUse = primitiveInitSize * (initSizes.Length > 0 ? initSizes[i] : 1);
			float3 randPos = positions[i];
			float3 position0 = randPos + new float3(UnityEngine.Random.onUnitSphere) * initSizeToUse;
			float3 position1 = randPos + new float3(UnityEngine.Random.onUnitSphere) * initSizeToUse;
			float3 position2 = randPos + new float3(UnityEngine.Random.onUnitSphere) * initSizeToUse;
			if (targetMode == TargetMode.Image)
			{
				randPos.z = i / (float)primitiveCount; // Unique Z per triangle
				position0 = new float3(-initSizeToUse, -initSizeToUse, 0.0f) + (new float3(UnityEngine.Random.value, UnityEngine.Random.value, 0.0f) * 2.0f - 1.0f) * initSizeToUse * 0.5f;
				position1 = new float3(0.0f, initSizeToUse, 0.0f) + (new float3(UnityEngine.Random.value, UnityEngine.Random.value, 0.0f) * 2.0f - 1.0f) * initSizeToUse * 0.5f;
				position2 = new float3(initSizeToUse, -initSizeToUse, 0.0f) + (new float3(UnityEngine.Random.value, UnityEngine.Random.value, 0.0f) * 2.0f - 1.0f) * initSizeToUse * 0.5f;
			}
			randData[i * primitiveFloatSize + offset + 0] = position0.x; randData[i * primitiveFloatSize + offset + 1] = position0.y; randData[i * primitiveFloatSize + offset + 2] = position0.z; offset += 3;
			randData[i * primitiveFloatSize + offset + 0] = position1.x; randData[i * primitiveFloatSize + offset + 1] = position1.y; randData[i * primitiveFloatSize + offset + 2] = position1.z; offset += 3;
			randData[i * primitiveFloatSize + offset + 0] = position2.x; randData[i * primitiveFloatSize + offset + 1] = position2.y; randData[i * primitiveFloatSize + offset + 2] = position2.z; offset += 3;

			// Alpha
			if (transparencyMode != TransparencyMode.None)
			{
				randData[i * primitiveFloatSize + offset + 0] = 1.0f; offset += 1;
			}

			// Color
			if (sphericalHarmonicsMode == SphericalHarmonicsMode.None) // else no need to do anything, all init to zero
			{
				for (int j = 0; j < (optimPrimitive == PrimitiveType.TrianglesGradientUnlit ? 3 : 1); j++)
				{
					Color randColor = UnityEngine.Random.ColorHSV(0, 1, 0, 1);
					float3 color = new float3(randColor.r, randColor.g, randColor.b);
					//float3 color = new float3(0.5f, 0.5f, 0.5f);
					//float3 color = new float3(0.0f, 0.0f, 0.0f);
					randData[i * primitiveFloatSize + offset + 0] = color.x; randData[i * primitiveFloatSize + offset + 1] = color.y; randData[i * primitiveFloatSize + offset + 2] = color.z; offset += 3;
				}
			}
		}

		// Upload data to GPU
		primitiveBufferToInit = new ComputeBuffer(primitiveCount, sizeof(float) * primitiveFloatSize);
		primitiveBufferToInit.SetData(randData);
	}

	public void InitEnvMapPrimitiveBuffer(ref ComputeBuffer primitiveBufferToInit)
	{
		float3[] randData = new float3[envMapResolution * envMapResolution];
		for (int i = 0; i < envMapResolution * envMapResolution; i++)
		{
			float3 newTexel;
			Color randColor = UnityEngine.Random.ColorHSV(0, 1, 0, 1);
			newTexel = new float3(randColor.r, randColor.g, randColor.b);
			//newTexel.color = new float3(1.0f, 1.0f, 1.0f);
			newTexel = new float3(0.5f, 0.5f, 0.5f);
			//newTexel.color = new float3(0.0f, 0.0f, 0.0f);
			randData[i] = newTexel;
		}
		primitiveBufferToInit = new ComputeBuffer(envMapResolution * envMapResolution, System.Runtime.InteropServices.Marshal.SizeOf(typeof(float3)));
		primitiveBufferToInit.SetData(randData);
	}




	// ======================= STUFF =======================
	public enum DisplayMode
	{
		Optimization,
		Target
	}

	public enum TargetMode
	{
		Image,
		Model,
		COLMAP
	}

	public enum PrimitiveType
	{
		TrianglesSolidUnlit,
		TrianglesGradientUnlit,
		TrianglesGaussianUnlit
	}

	public enum TransparencyMode
	{
		None,
		SortedAlpha,
		StochasticAlpha
	}

	public enum SphericalHarmonicsMode
	{
		None,
		TwoBands,
		ThreeBands,
		FourBands
	}

	public enum BackgroundMode
	{
		Color,
		RandomColor,
		EnvMap
	}

	public enum LossMode
	{
		L1,
		L2
	}

	public enum Optimizer
	{
		RMSProp,
		Adam
	}

	public enum ParameterOptimSeparationMode
	{
		None,
		GeometryAndAppearance,
		Full
	}
}
