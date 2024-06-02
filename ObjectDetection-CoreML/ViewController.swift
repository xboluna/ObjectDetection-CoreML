//
//  ViewController.swift
//  SSDMobileNet-CoreML
//
//  Created by GwakDoyoung on 01/02/2019.
//  Copyright © 2019 tucan9389. All rights reserved.
//

import UIKit
import Vision
import CoreMedia

class ViewController: UIViewController {

    // MARK: - UI Properties
    @IBOutlet weak var videoPreview: UIView!
    @IBOutlet weak var boxesView: DrawingBoundingBoxView!
    @IBOutlet weak var labelsTableView: UITableView!
    
    @IBOutlet weak var inferenceLabel: UILabel!
    @IBOutlet weak var etimeLabel: UILabel!
    @IBOutlet weak var fpsLabel: UILabel!
    
    // MARK - Core ML model
    // YOLOv3(iOS12+), YOLOv3FP16(iOS12+), YOLOv3Int8LUT(iOS12+)
    // YOLOv3Tiny(iOS12+), YOLOv3TinyFP16(iOS12+), YOLOv3TinyInt8LUT(iOS12+)
    // MobileNetV2_SSDLite(iOS12+), ObjectDetector(iOS12+)
    // yolov5n(iOS13+), yolov5s(iOS13+), yolov5m(iOS13+), yolov5l(iOS13+), yolov5x(iOS13+)
    // yolov5n6(iOS13+), yolov5s6(iOS13+), yolov5m6(iOS13+), yolov5l6(iOS13+), yolov5x6(iOS13+)
    // yolov8n(iOS14+), yolov8s(iOS14+), yolov8m(iOS14+), yolov8l(iOS14+), yolov8x(iOS14+)
    lazy var objectDetectionModel = { return try? yolov8s() }()
    
    // MARK: - Vision Properties
    var request: VNCoreMLRequest?
    var visionModel: VNCoreMLModel?
    var isInferencing = false
    
    // MARK: - Tracking Properties
    var trackedObjects = [UUID: VNDetectedObjectObservation]()
    var sequenceRequestHandler = VNSequenceRequestHandler()
    var currentImageBeingProcessed: CVPixelBuffer?

    // MARK: - AV Property
    var videoCapture: VideoCapture!
    let semaphore = DispatchSemaphore(value: 1)
    var lastExecution = Date()
    
    // MARK: - TableView Data
    var predictions: [VNRecognizedObjectObservation] = []
    var recognizedTexts = [UUID:String]()
    
    // MARK - Performance Measurement Property
    private let 👨‍🔧 = 📏()
    
    let maf1 = MovingAverageFilter()
    let maf2 = MovingAverageFilter()
    let maf3 = MovingAverageFilter()
    
    // MARK: - View Controller Life Cycle
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // setup the model
        setUpModel()
        
        // setup camera
        setUpCamera()
        
        // setup delegate for performance measurement
        👨‍🔧.delegate = self
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        self.videoCapture.start()
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        self.videoCapture.stop()
    }
    
    // MARK: - Setup Core ML
    func setUpModel() {
        guard let objectDetectionModel = objectDetectionModel else { fatalError("fail to load the model") }
        if let visionModel = try? VNCoreMLModel(for: objectDetectionModel.model) {
            self.visionModel = visionModel
            request = VNCoreMLRequest(model: visionModel, completionHandler: visionRequestDidComplete)
            request?.imageCropAndScaleOption = .scaleFill
        } else {
            fatalError("fail to create vision model")
        }
    }

    // MARK: - SetUp Video
    func setUpCamera() {
        videoCapture = VideoCapture()
        videoCapture.delegate = self
        videoCapture.fps = 30
        videoCapture.setUp(sessionPreset: .vga640x480) { success in
            
            if success {
                // add preview view on the layer
                if let previewLayer = self.videoCapture.previewLayer {
                    self.videoPreview.layer.addSublayer(previewLayer)
                    self.resizePreviewLayer()
                }
                
                // start video preview when setup is done
                self.videoCapture.start()
            }
        }
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        resizePreviewLayer()
    }
    
    func resizePreviewLayer() {
        videoCapture.previewLayer?.frame = videoPreview.bounds
    }
}

// MARK: - VideoCaptureDelegate
extension ViewController: VideoCaptureDelegate {
    func videoCapture(_ capture: VideoCapture, didCaptureVideoFrame pixelBuffer: CVPixelBuffer?, timestamp: CMTime) {
        // the captured image from camera is contained on pixelBuffer
        if !self.isInferencing, let pixelBuffer = pixelBuffer {
            self.isInferencing = true
            
            // start of measure
            self.👨‍🔧.🎬👏()

            self.predictAndTrack(pixelBuffer: pixelBuffer)
        }
    }
}

extension ViewController {
    func recognizeText(in image: CVPixelBuffer, for boundingBox: CGRect, completion: @escaping (String) -> Void) {
        let textRecognitionRequest = VNRecognizeTextRequest { (request, error) in 
            guard let observations = request.results as? [VNRecognizedTextObservation], error == nil else {
                completion("no observation") // TODO change
                return
            }

            // TODO grab multiple candidates to send as fuzzy search?
            let recognizedStrings = observations.compactMap { $0.topCandidates(1).first?.string }
        
            completion (recognizedStrings.joined(separator: " "))
        }

        textRecognitionRequest.recognitionLevel = .accurate
        textRecognitionRequest.usesLanguageCorrection = true // what does this do?

        let handler = VNImageRequestHandler(cvPixelBuffer: image, options: [:])
        do {
            try handler.perform([textRecognitionRequest])
        } catch {
            completion("error request") // TODO change
        }
    }

    func predictAndTrack(pixelBuffer: CVPixelBuffer) {
        guard let request = request else { fatalError() }
        // vision framework configures the input size of image following our model's input configuration automatically
        self.semaphore.wait()
        self.currentImageBeingProcessed = pixelBuffer
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
        try? handler.perform([request])
    }
    
    // MARK: - Post-processing
    func visionRequestDidComplete(request: VNRequest, error: Error?) {
        self.👨‍🔧.🏷(with: "endInference")

        guard let predictions = request.results as? [VNRecognizedObjectObservation] else {
            self.👨‍🔧.🎬🤚()
            self.isInferencing = false
            self.semaphore.signal()
            return
        }

        // keep track of new predictions
        var newTrackedObjects = [UUID: VNDetectedObjectObservation]()
        var newPredictions = [VNRecognizedObjectObservation]()

        for prediction in predictions {
            var matched = false
            for (uuid, trackedObject) in trackedObjects {
                // check if prediction matches existing tracked object.
                // TODO Let's change this to an IoU percentage
                //      rather than just whether it intersects at all.
                // In any case, this is very naive and worth replacing with a real
                //      deeptracker or similar at such a time that it is worth it.
                if trackedObject.boundingBox.intersects(prediction.boundingBox) {
                    newTrackedObjects[uuid] = prediction
                    matched = true
                    break
                }
            }

            if !matched {
                // add new prediction
                // TODO Perform OCR at this step.
                // TODO Ping cloud with OCR char results.
                let observation = VNDetectedObjectObservation(boundingBox: prediction.boundingBox)
                newTrackedObjects[observation.uuid] = observation
            }
            newPredictions.append(prediction)
        }
        
        // track new set of objects if image is available in this step
        if let pixelBuffer = currentImageBeingProcessed {
            trackObjects(pixelBuffer: pixelBuffer, newTrackedObjects: newTrackedObjects, newPredictions: newPredictions)
            
        }
    }

    func trackObjects(pixelBuffer: CVPixelBuffer, newTrackedObjects: [UUID: VNDetectedObjectObservation], newPredictions: [VNRecognizedObjectObservation]) {
        var trackingRequests = [VNRequest]()

        // recognize text within each bbox
        let dispatchGroup = DispatchGroup()

        for prediction in newPredictions {
            dispatchGroup.enter()
            let uuid = UUID()
            recognizeText(in: pixelBuffer, for: prediction.boundingBox) { text in
                self.recognizedTexts[uuid] = text
                dispatchGroup.leave()
            }
        }

        for (_, observation) in newTrackedObjects {
            let trackingRequest = VNTrackObjectRequest(detectedObjectObservation: observation)
            trackingRequests.append(trackingRequest)
        }

        do {
            try sequenceRequestHandler.perform(trackingRequests, on: pixelBuffer)

            var finalTrackedObjects = [UUID: VNDetectedObjectObservation]()

            for request in trackingRequests {
                if let results = request.results as? [VNDetectedObjectObservation], let result = results.first {
                    finalTrackedObjects[result.uuid] = result
                }
            }

            self.predictions = newPredictions
            self.trackedObjects = finalTrackedObjects

            DispatchQueue.main.async {
                self.boxesView.predictedObjects = self.predictions
                self.labelsTableView.reloadData()

                // end of measure
                self.👨‍🔧.🎬🤚()
                
                self.isInferencing = false
            }
        } catch {
            self.👨‍🔧.🎬🤚()
            self.isInferencing = false
        }

        self.semaphore.signal()

    }
}

extension ViewController: UITableViewDataSource {
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return predictions.count
    }
    
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        guard let cell = tableView.dequeueReusableCell(withIdentifier: "InfoCell") else {
            return UITableViewCell()
        }

        let rectString = predictions[indexPath.row].boundingBox.toString(digit: 2)
        let confidence = predictions[indexPath.row].labels.first?.confidence ?? -1
        let confidenceString = String(format: "%.3f", confidence/*Math.sigmoid(confidence)*/)
        
        cell.textLabel?.text = predictions[indexPath.row].label ?? "N/A"
        cell.detailTextLabel?.text = "\(rectString), \(confidenceString)"
        
        // display recognized text
        let uuid = predictions[indexPath.row].uuid
        if let recognizedText = recognizedTexts[uuid] {
            cell.detailTextLabel?.text = "\(cell.detailTextLabel?.text ?? ""), \(recognizedText)"
        }
        
        return cell
    }
}

// MARK: - 📏(Performance Measurement) Delegate
extension ViewController: 📏Delegate {
    func updateMeasure(inferenceTime: Double, executionTime: Double, fps: Int) {
        //print(executionTime, fps)
        DispatchQueue.main.async {
            self.maf1.append(element: Int(inferenceTime*1000.0))
            self.maf2.append(element: Int(executionTime*1000.0))
            self.maf3.append(element: fps)
            
            self.inferenceLabel.text = "inference: \(self.maf1.averageValue) ms"
            self.etimeLabel.text = "execution: \(self.maf2.averageValue) ms"
            self.fpsLabel.text = "fps: \(self.maf3.averageValue)"
        }
    }
}

class MovingAverageFilter {
    private var arr: [Int] = []
    private let maxCount = 10
    
    public func append(element: Int) {
        arr.append(element)
        if arr.count > maxCount {
            arr.removeFirst()
        }
    }
    
    public var averageValue: Int {
        guard !arr.isEmpty else { return 0 }
        let sum = arr.reduce(0) { $0 + $1 }
        return Int(Double(sum) / Double(arr.count))
    }
}


