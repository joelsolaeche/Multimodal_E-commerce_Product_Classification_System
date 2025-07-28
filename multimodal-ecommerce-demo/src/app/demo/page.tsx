'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  CloudArrowUpIcon,
  PhotoIcon,
  SparklesIcon,
  ChartBarIcon,
  EyeIcon,
  ChatBubbleBottomCenterTextIcon
} from '@heroicons/react/24/outline';
import Image from 'next/image';
import Link from 'next/link';

type TabType = 'text' | 'image' | 'multimodal';

interface Prediction {
  category: string;
  name: string;  
  confidence: number;
}

interface ClassificationResult {
  predictions: Prediction[];
  model_used: string;
  method: string;
  text?: string;
  image_data?: string;
  has_image?: boolean;
}

export default function DemoPage() {
  const [activeTab, setActiveTab] = useState<TabType>('multimodal');
  const [text, setText] = useState('');
  const [image, setImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string>('');
  const [results, setResults] = useState<ClassificationResult | null>(null);
  const [loading, setLoading] = useState(false);
  
  // Use environment variable or fallback to Railway URL
  // Ensure URL has no trailing slash to prevent double-slash issues
  const API_URL = process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, '');

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setImage(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const classifyText = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/api/classify/text`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });
      
      if (!response.ok) {
        throw new Error(`Server responded with status: ${response.status}`);
      }
      
      const result = await response.json();
      setResults(result);
    } catch (error) {
      console.error('Error classifying text', error);
    } finally {
      setLoading(false);
    }
  };

  const classifyImage = async () => {
    if (!image) return;
    
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', image);
      
      const response = await fetch(`${API_URL}/api/classify/image`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Server responded with status: ${response.status}`);
      }
      
      const result = await response.json();
      setResults(result);
    } catch (error) {
      console.error('Error classifying image', error);
    } finally {
      setLoading(false);
    }
  };

  const classifyMultimodal = async () => {
    setLoading(true);
    try {
      const payload: { text: string; image_data?: string } = { text };
      
      if (image) {
        const reader = new FileReader();
        reader.onload = async (e) => {
          payload.image_data = e.target?.result as string;
          
          const response = await fetch(`${API_URL}/api/classify/multimodal`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload),
          });
          
          if (!response.ok) {
            throw new Error(`Server responded with status: ${response.status}`);
          }
          
          const result = await response.json();
          setResults(result);
          setLoading(false);
        };
        reader.readAsDataURL(image);
      } else {
        const response = await fetch(`${API_URL}/api/classify/multimodal`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(payload),
        });
        
        if (!response.ok) {
          throw new Error(`Server responded with status: ${response.status}`);
        }
        
        const result = await response.json();
        setResults(result);
        setLoading(false);
      }
    } catch (error) {
      console.error('Error in multimodal classification', error);
      setLoading(false);
    }
  };

  const handleClassify = async () => {
    if (activeTab === 'text' && text) {
      await classifyText();
    } else if (activeTab === 'image' && image) {
      await classifyImage();
    } else if (activeTab === 'multimodal' && (text || image)) {
      await classifyMultimodal();
    }
  };

  const tabs = [
    {
      id: 'multimodal' as TabType,
      name: 'Multimodal',
      icon: SparklesIcon,
      description: 'Combine text and image for best accuracy',
      color: 'from-purple-600 to-pink-600'
    },
    {
      id: 'text' as TabType,
      name: 'Text Only',
      icon: ChatBubbleBottomCenterTextIcon,
      description: 'Classify using product descriptions',
      color: 'from-blue-600 to-cyan-600'
    },
    {
      id: 'image' as TabType,
      name: 'Image Only',
      icon: EyeIcon,
      description: 'Classify using computer vision',
      color: 'from-green-600 to-emerald-600'
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      {/* Navigation */}
      <nav className="bg-white/80 backdrop-blur-md border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <Link href="/" className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
                <ChartBarIcon className="w-5 h-5 text-white" />
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
            Interactive <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">Classification</span>
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Test our multimodal AI system with your own data. Upload images, enter product descriptions, or combine both for the most accurate results.
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-12">
          {/* Input Section */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="space-y-8"
          >
            {/* Tab Selection */}
            <div className="bg-white/60 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
              <h2 className="text-2xl font-bold text-gray-800 mb-6">Choose Classification Mode</h2>
              <div className="space-y-4">
                {tabs.map((tab) => (
                  <motion.button
                    key={tab.id}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => setActiveTab(tab.id)}
                    className={`w-full p-4 rounded-xl border-2 transition-all duration-300 ${
                      activeTab === tab.id
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-200 bg-white hover:border-gray-300'
                    }`}
                  >
                    <div className="flex items-center space-x-4">
                      <div className={`w-12 h-12 rounded-xl bg-gradient-to-r ${tab.color} flex items-center justify-center`}>
                        <tab.icon className="w-6 h-6 text-white" />
                      </div>
                      <div className="text-left">
                        <h3 className="font-semibold text-gray-800">{tab.name}</h3>
                        <p className="text-sm text-gray-600">{tab.description}</p>
                      </div>
                    </div>
                  </motion.button>
                ))}
              </div>
            </div>

            {/* Text Input */}
            {(activeTab === 'text' || activeTab === 'multimodal') && (
              <div className="bg-white/60 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
                <h3 className="text-xl font-bold text-gray-800 mb-4">Product Description</h3>
                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="Enter a product description, e.g., 'Wireless Bluetooth headphones with noise cancellation'"
                  className="w-full h-32 p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                />
              </div>
            )}

            {/* Image Upload */}
            {(activeTab === 'image' || activeTab === 'multimodal') && (
              <div className="bg-white/60 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
                <h3 className="text-xl font-bold text-gray-800 mb-4">Product Image</h3>
                <div className="space-y-4">
                  <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100">
                    <div className="flex flex-col items-center justify-center pt-5 pb-6">
                      <CloudArrowUpIcon className="w-8 h-8 mb-4 text-gray-500" />
                      <p className="mb-2 text-sm text-gray-500">
                        <span className="font-semibold">Click to upload</span> or drag and drop
                      </p>
                      <p className="text-xs text-gray-500">PNG, JPG or JPEG</p>
                    </div>
                    <input
                      type="file"
                      className="hidden"
                      accept="image/*"
                      onChange={handleImageUpload}
                    />
                  </label>
                  
                  {imagePreview && (
                    <div className="relative">
                      <Image
                        src={imagePreview}
                        alt="Preview"
                        width={200}
                        height={200}
                        className="w-full h-48 object-cover rounded-lg"
                      />
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Classify Button */}
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleClassify}
              disabled={loading || (!text && !image)}
              className="w-full py-4 px-6 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <div className="flex items-center justify-center space-x-2">
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                  <span>Classifying...</span>
                </div>
              ) : (
                <div className="flex items-center justify-center space-x-2">
                  <SparklesIcon className="w-5 h-5" />
                  <span>Classify Product</span>
                </div>
              )}
            </motion.button>
          </motion.div>

          {/* Results Section */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="space-y-8"
          >
            <div className="bg-white/60 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
              <h2 className="text-2xl font-bold text-gray-800 mb-6">Classification Results</h2>
              
              {!results ? (
                <div className="text-center py-12">
                  <PhotoIcon className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-500">Upload an image or enter text to see results</p>
                </div>
              ) : (
                <div className="space-y-6">
                  {/* Model Info */}
                  <div className="bg-blue-50 rounded-lg p-4">
                    <h3 className="font-semibold text-blue-800 mb-1">Model Used</h3>
                    <p className="text-blue-600 text-sm">{results.model_used}</p>
                  </div>

                  {/* Predictions */}
                  <div>
                    <h3 className="font-semibold text-gray-800 mb-4">Top Predictions</h3>
                    <div className="space-y-3">
                      {results.predictions.map((prediction, index) => (
                        <div
                          key={index}
                          className="flex items-center justify-between p-4 bg-white rounded-lg border border-gray-200"
                        >
                          <div>
                            <h4 className="font-medium text-gray-800">{prediction.name}</h4>
                            <p className="text-sm text-gray-500">Category: {prediction.category}</p>
                          </div>
                          <div className="text-right">
                            <div className="text-lg font-bold text-blue-600">
                              {(prediction.confidence * 100).toFixed(1)}%
                            </div>
                            <div className="w-20 bg-gray-200 rounded-full h-2 mt-1">
                              <div
                                className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full"
                                style={{ width: `${prediction.confidence * 100}%` }}
                              ></div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Image Preview */}
                  {results.image_data && (
                    <div>
                      <h3 className="font-semibold text-gray-800 mb-4">Analyzed Image</h3>
                      <Image
                        src={results.image_data}
                        alt="Analyzed product"
                        width={300}
                        height={300}
                        className="w-full h-48 object-cover rounded-lg"
                      />
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Demo Note */}
            <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
              <h3 className="font-semibold text-amber-800 mb-2">Demo Note</h3>
              <p className="text-amber-700 text-sm">
                This demo uses real TF-IDF similarity search and embedding-based classification. 
                Results are based on actual product data analysis and machine learning models.
              </p>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
} 