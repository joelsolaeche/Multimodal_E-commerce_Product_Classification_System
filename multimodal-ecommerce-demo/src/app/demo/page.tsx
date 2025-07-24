'use client';

import { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import {
  PhotoIcon,
  ChatBubbleBottomCenterTextIcon,
  SparklesIcon,
  ArrowPathIcon,
  CloudArrowUpIcon,
  DocumentTextIcon,
  CheckCircleIcon,
  XMarkIcon
} from '@heroicons/react/24/outline';
import Link from 'next/link';

interface Prediction {
  category: string;
  name: string;
  confidence: number;
}

interface ClassificationResult {
  predictions: Prediction[];
  model_used: string;
  text?: string;
  image_data?: string;
  has_image?: boolean;
}

export default function DemoPage() {
  const [activeTab, setActiveTab] = useState<'text' | 'image' | 'multimodal'>('text');
  const [textInput, setTextInput] = useState('');
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<ClassificationResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      setImageFile(file);
      const reader = new FileReader();
      reader.onload = () => {
        setImagePreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif', '.webp']
    },
    multiple: false
  });

  const classifyText = async () => {
    if (!textInput.trim()) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('http://localhost:8000/api/classify/text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: textInput })
      });
      
      if (!response.ok) throw new Error('Classification failed');
      
      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError('Failed to classify text. Make sure the API server is running.');
    } finally {
      setLoading(false);
    }
  };

  const classifyImage = async () => {
    if (!imageFile) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('file', imageFile);
      
      const response = await fetch('http://localhost:8000/api/classify/image', {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) throw new Error('Classification failed');
      
      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError('Failed to classify image. Make sure the API server is running.');
    } finally {
      setLoading(false);
    }
  };

  const classifyMultimodal = async () => {
    if (!textInput.trim() && !imageFile) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const payload: any = { text: textInput };
      
      if (imageFile) {
        const reader = new FileReader();
        reader.onload = async () => {
          payload.image_data = reader.result as string;
          
          const response = await fetch('http://localhost:8000/api/classify/multimodal', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          });
          
          if (!response.ok) throw new Error('Classification failed');
          
          const data = await response.json();
          setResults(data);
          setLoading(false);
        };
        reader.readAsDataURL(imageFile);
      } else {
        const response = await fetch('http://localhost:8000/api/classify/multimodal', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        
        if (!response.ok) throw new Error('Classification failed');
        
        const data = await response.json();
        setResults(data);
        setLoading(false);
      }
    } catch (err) {
      setError('Failed to perform multimodal classification. Make sure the API server is running.');
      setLoading(false);
    }
  };

  const handleClassify = () => {
    switch (activeTab) {
      case 'text':
        classifyText();
        break;
      case 'image':
        classifyImage();
        break;
      case 'multimodal':
        classifyMultimodal();
        break;
    }
  };

  const clearAll = () => {
    setTextInput('');
    setImageFile(null);
    setImagePreview(null);
    setResults(null);
    setError(null);
  };

  const sampleTexts = [
    "Sony Alpha 7R IV full-frame mirrorless camera with 61MP sensor and in-body stabilization",
    "Samsung 55-inch 4K UHD Smart TV with HDR and built-in streaming apps",
    "Apple MacBook Pro 16-inch with M2 Pro chip, 32GB RAM, and 1TB SSD storage",
    "Bose QuietComfort 45 wireless headphones with active noise cancellation"
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      {/* Navigation */}
      <nav className="bg-white/80 backdrop-blur-md border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <Link href="/" className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
                <SparklesIcon className="w-5 h-5 text-white" />
              </div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                Interactive Demo
              </h1>
            </Link>
            <Link 
              href="/"
              className="text-gray-600 hover:text-blue-600 transition-colors"
            >
              ← Back to Home
            </Link>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-12"
        >
          <h1 className="text-4xl md:text-5xl font-bold text-gray-800 mb-6">
            Test the <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">AI System</span>
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Try our multimodal classification system with your own text descriptions or product images
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-12">
          {/* Input Section */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="space-y-6"
          >
            {/* Tab Selection */}
            <div className="bg-white/60 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
              <h2 className="text-2xl font-bold text-gray-800 mb-6">Classification Mode</h2>
              <div className="flex flex-wrap gap-4">
                {[
                  { key: 'text', label: 'Text Only', icon: DocumentTextIcon, color: 'from-blue-600 to-cyan-600' },
                  { key: 'image', label: 'Image Only', icon: PhotoIcon, color: 'from-purple-600 to-pink-600' },
                  { key: 'multimodal', label: 'Multimodal', icon: SparklesIcon, color: 'from-green-600 to-emerald-600' }
                ].map((tab) => (
                  <button
                    key={tab.key}
                    onClick={() => setActiveTab(tab.key as any)}
                    className={`flex items-center space-x-2 px-6 py-3 rounded-xl font-semibold transition-all duration-300 ${
                      activeTab === tab.key
                        ? `bg-gradient-to-r ${tab.color} text-white shadow-lg`
                        : 'bg-white/60 text-gray-700 hover:bg-white/80'
                    }`}
                  >
                    <tab.icon className="w-5 h-5" />
                    <span>{tab.label}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Text Input */}
            {(activeTab === 'text' || activeTab === 'multimodal') && (
              <div className="bg-white/60 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
                <div className="flex items-center space-x-3 mb-4">
                  <ChatBubbleBottomCenterTextIcon className="w-6 h-6 text-blue-600" />
                  <h3 className="text-xl font-bold text-gray-800">Product Description</h3>
                </div>
                <textarea
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  placeholder="Enter a product description..."
                  className="w-full h-32 p-4 border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none bg-white/80 backdrop-blur-sm"
                />
                <div className="mt-4">
                  <p className="text-sm text-gray-600 mb-2">Try these examples:</p>
                  <div className="flex flex-wrap gap-2">
                    {sampleTexts.map((sample, index) => (
                      <button
                        key={index}
                        onClick={() => setTextInput(sample)}
                        className="text-xs bg-blue-50 text-blue-700 px-3 py-1 rounded-full hover:bg-blue-100 transition-colors"
                      >
                        Example {index + 1}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Image Upload */}
            {(activeTab === 'image' || activeTab === 'multimodal') && (
              <div className="bg-white/60 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
                <div className="flex items-center space-x-3 mb-4">
                  <PhotoIcon className="w-6 h-6 text-purple-600" />
                  <h3 className="text-xl font-bold text-gray-800">Product Image</h3>
                </div>
                <div
                  {...getRootProps()}
                  className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-300 ${
                    isDragActive
                      ? 'border-purple-400 bg-purple-50'
                      : 'border-gray-300 hover:border-purple-400 hover:bg-purple-50/50'
                  }`}
                >
                  <input {...getInputProps()} />
                  {imagePreview ? (
                    <div className="space-y-4">
                      <img
                        src={imagePreview}
                        alt="Preview"
                        className="max-w-full max-h-48 mx-auto rounded-lg shadow-md"
                      />
                      <div className="flex justify-center space-x-4">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setImageFile(null);
                            setImagePreview(null);
                          }}
                          className="text-red-600 hover:text-red-700 flex items-center space-x-1"
                        >
                          <XMarkIcon className="w-4 h-4" />
                          <span>Remove</span>
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <CloudArrowUpIcon className="w-16 h-16 text-gray-400 mx-auto" />
                      <div>
                        <p className="text-lg font-medium text-gray-700">
                          {isDragActive ? 'Drop the image here' : 'Drag & drop an image here'}
                        </p>
                        <p className="text-gray-500">or click to select from your device</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex space-x-4">
              <button
                onClick={handleClassify}
                disabled={loading || (activeTab === 'text' && !textInput.trim()) || (activeTab === 'image' && !imageFile) || (activeTab === 'multimodal' && !textInput.trim() && !imageFile)}
                className="flex-1 bg-gradient-to-r from-blue-600 to-purple-600 text-white px-8 py-4 rounded-xl font-semibold text-lg hover:shadow-lg transition-all duration-300 flex items-center justify-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? (
                  <>
                    <ArrowPathIcon className="w-5 h-5 animate-spin" />
                    <span>Classifying...</span>
                  </>
                ) : (
                  <>
                    <SparklesIcon className="w-5 h-5" />
                    <span>Classify Product</span>
                  </>
                )}
              </button>
              <button
                onClick={clearAll}
                className="bg-white/80 backdrop-blur-sm text-gray-700 px-6 py-4 rounded-xl font-semibold hover:bg-white transition-all duration-300 border border-gray-200"
              >
                Clear All
              </button>
            </div>
          </motion.div>

          {/* Results Section */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="space-y-6"
          >
            <div className="bg-white/60 backdrop-blur-sm rounded-2xl p-8 border border-white/20 min-h-[500px]">
              <h2 className="text-2xl font-bold text-gray-800 mb-6">Classification Results</h2>
              
              {error && (
                <div className="bg-red-50 border border-red-200 rounded-xl p-6 text-center">
                  <XMarkIcon className="w-12 h-12 text-red-500 mx-auto mb-4" />
                  <p className="text-red-700 font-medium">{error}</p>
                  <p className="text-red-600 text-sm mt-2">
                    Make sure the FastAPI server is running on http://localhost:8000
                  </p>
                </div>
              )}

              {!results && !error && !loading && (
                <div className="text-center text-gray-500 py-16">
                  <SparklesIcon className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                  <p className="text-lg font-medium">No results yet</p>
                  <p>Upload content and click "Classify Product" to see AI predictions</p>
                </div>
              )}

              {results && (
                <div className="space-y-6">
                  {/* Model Used */}
                  <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-xl p-4 border border-blue-200">
                    <div className="flex items-center space-x-2">
                      <CheckCircleIcon className="w-5 h-5 text-green-600" />
                      <span className="font-semibold text-gray-800">Model: {results.model_used}</span>
                    </div>
                  </div>

                  {/* Predictions */}
                  <div className="space-y-4">
                    <h3 className="text-lg font-semibold text-gray-800">Top Predictions</h3>
                    {results.predictions.map((prediction, index) => (
                      <div
                        key={index}
                        className="bg-white/80 rounded-xl p-4 border border-gray-200 hover:shadow-md transition-all duration-300"
                      >
                        <div className="flex justify-between items-start mb-2">
                          <div className="flex-1">
                            <h4 className="font-semibold text-gray-800">{prediction.name}</h4>
                            <p className="text-sm text-gray-600">Category: {prediction.category}</p>
                          </div>
                          <div className="text-right">
                            <div className="text-2xl font-bold text-blue-600">
                              {(prediction.confidence * 100).toFixed(1)}%
                            </div>
                            <div className="text-xs text-gray-500">confidence</div>
                          </div>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full transition-all duration-1000"
                            style={{ width: `${prediction.confidence * 100}%` }}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Input Summary */}
                  {(results.text || results.has_image) && (
                    <div className="bg-gray-50 rounded-xl p-4 border border-gray-200">
                      <h4 className="font-semibold text-gray-800 mb-2">Input Summary</h4>
                      {results.text && (
                        <p className="text-sm text-gray-600 mb-2">
                          <strong>Text:</strong> {results.text.substring(0, 100)}...
                        </p>
                      )}
                      {results.has_image && (
                        <p className="text-sm text-gray-600">
                          <strong>Image:</strong> Uploaded and processed
                        </p>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
} 