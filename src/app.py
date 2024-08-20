from flask import Flask, request, jsonify
import base64
import traceback
from time import time
import tempfile
import os
import yaml
import Setting
from model import Model
from script.script_writer import LocatorWriter, ScriptWriter
from script.strategy.strategy import Strategy
from script.strategy.transition_matching_strategy import TransitionMatchingStrategy
from script.test_case import TestCaseParser
from gensim.models import FastText
from gensim.test.utils import common_texts

app = Flask(__name__)

# Preload the model

with app.app_context():
    global model
    print("Loading model...")
    try:
        model = Model(Setting.MODEL)
        
    except Exception as e:
        model = FastText(vector_size=4, window=3, min_count=1) 
        model.build_vocab(corpus_iterable=common_texts)
        model.train(corpus_iterable=common_texts, total_examples=len(common_texts), epochs=10)
        print(e)
    print("Model loaded.")

@app.route('/generate-script', methods=['POST'])
def generate_script():
    start = time()
    parser = TestCaseParser()

    try:
        # Get the encoded test cases from the form field
        encoded_test_cases = request.form['test_cases']
        # Decode the Base64 encoded data
        test_cases_yaml = base64.b64decode(encoded_test_cases.encode()).decode()

        # Write the YAML data to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.yaml') as temp_file:
            temp_file.write(test_cases_yaml.encode())
            temp_file_path = temp_file.name
        
        # Load test cases from the temporary file
        test_cases = parser.load_file(temp_file_path)
        os.remove(temp_file_path)  # Clean up the temporary file
        model_load_time = time() - start
        print("start generation")

        script = ""
        for test_case in test_cases:
            print("--------------------")
            print(test_case.name)
            print("--------------------")
            strategy: Strategy = TransitionMatchingStrategy(model)
            method_string = strategy.to_code(test_case)
            LocatorWriter.end_testcase()
            script += "\n"
            script += method_string
        
        writer = ScriptWriter()
        generation_time = time() - start - model_load_time
        script = writer.write(script, generation_time, model_load_time)
        

        return jsonify({"status": "success", "script": script})

    except Exception as e:
        print(e)
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

    finally:
        elapsed_time = time() - start
        print(f"elapsed_time:{elapsed_time}[sec]")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

