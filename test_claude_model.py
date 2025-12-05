#!/usr/bin/env python3
"""
Test script to validate Claude API model availability
Tests the claude-3-opus-20240229 model to ensure no 404 errors
"""

import os
import sys

def test_claude_model():
    """Test if Claude model is accessible"""
    
    # Check if anthropic package is available
    try:
        import anthropic
        print("✅ anthropic package is installed")
    except ImportError:
        print("❌ anthropic package NOT installed")
        print("   Install with: pip install anthropic")
        return False
    
    # Check for API key
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ ANTHROPIC_API_KEY environment variable not set")
        print("   Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        return False
    
    print(f"✅ ANTHROPIC_API_KEY is set (length: {len(api_key)} chars)")
    
    # Test the model
    model_name = "claude-3-opus-20240229"
    print(f"\n🧪 Testing model: {model_name}")
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        # Simple test message
        message = client.messages.create(
            model=model_name,
            max_tokens=10,
            temperature=0.1,
            messages=[
                {"role": "user", "content": "Say 'OK' if you can read this."}
            ]
        )
        
        response_text = message.content[0].text
        print(f"✅ Model responded successfully!")
        print(f"   Response: {response_text}")
        print(f"   Model: {message.model}")
        print(f"   Tokens used: {message.usage.input_tokens} in, {message.usage.output_tokens} out")
        
        return True
        
    except anthropic.NotFoundError as e:
        print(f"❌ Model NOT FOUND (404 error)")
        print(f"   Error: {e}")
        print(f"   Model '{model_name}' is not available with this API key")
        return False
        
    except anthropic.AuthenticationError as e:
        print(f"❌ Authentication failed")
        print(f"   Error: {e}")
        print(f"   Check your ANTHROPIC_API_KEY")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}")
        print(f"   Error: {e}")
        return False

def test_alternative_models():
    """Test alternative Claude models if opus fails"""
    
    try:
        import anthropic
    except ImportError:
        return
    
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        return
    
    print("\n" + "="*60)
    print("Testing alternative Claude models...")
    print("="*60)
    
    alternative_models = [
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620",
    ]
    
    client = anthropic.Anthropic(api_key=api_key)
    
    for model in alternative_models:
        print(f"\n🧪 Testing: {model}")
        try:
            message = client.messages.create(
                model=model,
                max_tokens=5,
                temperature=0.1,
                messages=[{"role": "user", "content": "Hi"}]
            )
            print(f"   ✅ WORKS - Response: {message.content[0].text}")
        except anthropic.NotFoundError:
            print(f"   ❌ NOT FOUND (404)")
        except Exception as e:
            print(f"   ❌ Error: {type(e).__name__}")

if __name__ == "__main__":
    print("="*60)
    print("Claude API Model Test Script")
    print("="*60)
    print()
    
    success = test_claude_model()
    
    if not success:
        print("\n⚠️  Primary model test failed. Testing alternatives...")
        test_alternative_models()
        sys.exit(1)
    else:
        print("\n" + "="*60)
        print("✅ SUCCESS - Model is ready for production use!")
        print("="*60)
        sys.exit(0)
