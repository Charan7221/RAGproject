#!/usr/bin/env python3
"""
List all available Google Gemini models for your API key
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

try:
    import google.generativeai as genai
    
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY or GEMINI_API_KEY not found in environment")
        print("Please set your API key in the .env file")
        sys.exit(1)
    
    genai.configure(api_key=api_key)
    
    print("🔍 Fetching available models...\n")
    print("=" * 80)
    print(f"{'Model Name':<40} {'Supports':<40}")
    print("=" * 80)
    
    generation_models = []
    
    for model in genai.list_models():
        model_name = model.name.replace("models/", "")
        supports = []
        
        if 'generateContent' in model.supported_generation_methods:
            supports.append("generateContent")
            generation_models.append(model_name)
        if 'embedContent' in model.supported_generation_methods:
            supports.append("embedContent")
        
        if supports:
            print(f"{model_name:<40} {', '.join(supports):<40}")
    
    print("=" * 80)
    print(f"\n✅ Found {len(generation_models)} models that support text generation:")
    for model in generation_models:
        print(f"   - {model}")
    
    print("\n📝 Recommended models for your RAG application:")
    recommended = [m for m in generation_models if 'pro' in m.lower() or 'flash' in m.lower()]
    if recommended:
        for model in recommended[:5]:  # Show top 5
            print(f"   ✓ {model}")
    else:
        print("   ✓ Use any model from the list above that supports generateContent")
    
    print("\n💡 To use a model, update config/config.yml:")
    print("   llm:")
    print(f"     model: \"{generation_models[0] if generation_models else 'gemini-pro'}\"")
    
except ImportError:
    print("❌ Error: google-generativeai package not installed")
    print("Install it with: pip install google-generativeai")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
