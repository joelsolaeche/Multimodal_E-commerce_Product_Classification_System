'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  ChartBarIcon,
  CpuChipIcon,
  EyeIcon,
  ChatBubbleBottomCenterTextIcon,
  PhotoIcon,
  ArrowRightIcon,
  CheckCircleIcon,
  CubeIcon,
  LightBulbIcon
} from '@heroicons/react/24/outline';
import Link from 'next/link';

interface Stats {
  total_products: number;
  total_categories: number;
  total_images: number;
  has_embeddings: boolean;
}

export default function HomePage() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);
  
  // Use environment variable or fallback to Railway API URL
  // Ensure URL has no trailing slash to prevent double-slash issues
  const API_URL = process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, '');

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const response = await fetch(`${API_URL}/api/stats`);
      
      if (!response.ok) {
        throw new Error(`Server responded with status: ${response.status}`);
      }
      
      const data = await response.json();
      setStats(data);
    } catch (error) {
      console.error('Failed to fetch stats:', error);
    } finally {
      setLoading(false);
    }
  };

  const achievements = [
    { label: "Classification Accuracy", value: "85%+", icon: ChartBarIcon },
    { label: "Product Images", value: stats?.total_images?.toLocaleString() || "49K+", icon: PhotoIcon },
    { label: "Product Categories", value: stats?.total_categories?.toString() || "100+", icon: CubeIcon },
    { label: "ML Models Implemented", value: "10+", icon: CpuChipIcon }
  ];

  const features = [
    {
      title: "Computer Vision Models",
      description: "ResNet50/101, DenseNet, ConvNeXt V2, Vision Transformer, Swin Transformer",
      icon: EyeIcon,
      color: "from-blue-600 to-cyan-600"
    },
    {
      title: "Natural Language Processing",
      description: "Sentence-BERT (MiniLM), Transformer Embeddings, OpenAI API Integration",
      icon: ChatBubbleBottomCenterTextIcon,
      color: "from-purple-600 to-pink-600"
    },
    {
      title: "Multimodal Fusion",
      description: "Advanced ML approaches combining visual and textual features",
      icon: LightBulbIcon,
      color: "from-green-600 to-emerald-600"
    },
    {
      title: "Production Ready",
      description: "Docker support, comprehensive testing, deployment documentation",
      icon: CheckCircleIcon,
      color: "from-orange-600 to-red-600"
    }
  ];

  const models = [
    "ResNet50 / ResNet101",
    "DenseNet121 / DenseNet169", 
    "ConvNeXt V2 (Tiny, Base, Large)",
    "Vision Transformer (ViT)",
    "Swin Transformer",
    "Sentence-BERT (MiniLM)",
    "Random Forest Classifier",
    "Logistic Regression",
    "Custom MLP Networks",
    "Multimodal Fusion Models"
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      {/* Navigation */}
      <nav className="bg-white/80 backdrop-blur-md border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
                <CpuChipIcon className="w-5 h-5 text-white" />
              </div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                Multimodal E-commerce AI
              </h1>
            </div>
            <div className="hidden md:flex space-x-8">
              <Link href="/demo" className="text-gray-700 hover:text-blue-600 transition-colors">
                Demo
              </Link>
              <Link href="/models" className="text-gray-700 hover:text-blue-600 transition-colors">  
                Models
              </Link>
              <Link href="/performance" className="text-gray-700 hover:text-blue-600 transition-colors">
                Performance
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-20 pb-16 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <h1 className="text-5xl md:text-7xl font-bold mb-6">
              <span className="bg-gradient-to-r from-blue-600 via-purple-600 to-teal-600 bg-clip-text text-transparent">
                Multimodal AI
              </span>
              <br />
              <span className="text-gray-800">Product Classification</span>
            </h1>
            <p className="text-xl md:text-2xl text-gray-600 mb-8 max-w-4xl mx-auto leading-relaxed">
              State-of-the-art machine learning system combining{' '}
              <span className="font-semibold text-blue-600">computer vision</span> and{' '}
              <span className="font-semibold text-purple-600">natural language processing</span>{' '}
              for automated e-commerce product categorization
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link 
                href="/demo"
                className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-8 py-4 rounded-xl font-semibold text-lg hover:shadow-lg transition-all duration-300 flex items-center justify-center group"
              >
                Try Live Demo
                <ArrowRightIcon className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" />
              </Link>
              <Link 
                href="/models"
                className="bg-white/80 backdrop-blur-sm text-gray-800 px-8 py-4 rounded-xl font-semibold text-lg border border-gray-200 hover:shadow-lg transition-all duration-300"
              >
                Explore Models
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {achievements.map((achievement, index) => (
              <motion.div
                key={achievement.label}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="bg-white/60 backdrop-blur-sm rounded-2xl p-6 text-center border border-white/20 hover:bg-white/80 transition-all duration-300"
              >
                <achievement.icon className="w-8 h-8 text-blue-600 mx-auto mb-3" />
                <div className="text-3xl font-bold text-gray-800 mb-1">
                  {loading ? "..." : achievement.value}
                </div>
                <div className="text-sm text-gray-600">{achievement.label}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="models" className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold text-gray-800 mb-6">
              Technical Highlights
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Advanced machine learning architectures and production-ready implementation
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 gap-8 mb-16">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, x: index % 2 === 0 ? -20 : 20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="bg-white/60 backdrop-blur-sm rounded-2xl p-8 border border-white/20 hover:bg-white/80 transition-all duration-300"
              >
                <div className={`w-12 h-12 rounded-xl bg-gradient-to-r ${feature.color} flex items-center justify-center mb-6`}>
                  <feature.icon className="w-6 h-6 text-white" />
                </div>
                <h3 className="text-2xl font-bold text-gray-800 mb-4">{feature.title}</h3>
                <p className="text-gray-600 leading-relaxed">{feature.description}</p>
              </motion.div>
            ))}
          </div>

          {/* Models Grid */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
            className="bg-white/60 backdrop-blur-sm rounded-2xl p-8 border border-white/20"
          >
            <h3 className="text-2xl font-bold text-gray-800 mb-6 text-center">
              Implemented ML Models & Architectures
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
              {models.map((model, index) => (
                <motion.div
                  key={model}
                  initial={{ opacity: 0, scale: 0.9 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.4, delay: index * 0.05 }}
                  className="bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-200 rounded-lg p-3 text-center hover:shadow-md transition-all duration-300"
                >
                  <div className="text-sm font-medium text-gray-800">{model}</div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
            className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl p-12 text-white"
          >
            <h2 className="text-3xl md:text-4xl font-bold mb-6">
              Experience the Power of Multimodal AI
            </h2>
            <p className="text-xl mb-8 opacity-90">
              Try our interactive demo to see how computer vision and NLP work together 
              for superior product classification accuracy
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link 
                href="/demo"
                className="bg-white text-blue-600 px-8 py-4 rounded-xl font-semibold text-lg hover:shadow-lg transition-all duration-300 flex items-center justify-center group"
              >
                Launch Interactive Demo
                <ArrowRightIcon className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" />
              </Link>
              <Link 
                href="/performance"
                className="bg-white/10 backdrop-blur-sm text-white px-8 py-4 rounded-xl font-semibold text-lg border border-white/20 hover:bg-white/20 transition-all duration-300"
              >
                View Performance Metrics
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-white/80 backdrop-blur-sm border-t border-gray-200 py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto text-center">
          <div className="flex items-center justify-center space-x-3 mb-6">
            <div className="w-8 h-8 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
              <CpuChipIcon className="w-5 h-5 text-white" />
            </div>
            <h3 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Multimodal E-commerce AI
            </h3>
          </div>
          <p className="text-gray-600">
            Advanced machine learning portfolio demonstrating multimodal AI expertise
          </p>
        </div>
      </footer>
    </div>
  );
}
