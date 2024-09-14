from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn

app = Flask(__name__)

# Paths
MODEL_PATH = "/home/janani/Documents/pivony/archive/output/MLTC_model_state.bin"
FEEDBACK_FILE = "feedback.tsv"

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define the custom BERT model class
class BERTClass(nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(768, 8)  # Assuming 8 output classes

    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids
        )
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output

# Load the trained model
model = BERTClass()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')), strict=False)
model.eval()

# Function to save data to the TSV file with appending
def save_to_tsv(data):
    df = pd.DataFrame(data, columns=['title', 'main_categories'])
    try:
        with open(FEEDBACK_FILE, 'a') as f:
            df.to_csv(f, sep='\t', header=f.tell() == 0, index=False)
    except Exception as e:
        print(f"Error saving to TSV: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_data', methods=['POST'])
def upload_data():
    title_abstract = request.form['title_abstract']
    main_categories = request.form['main_categories']
    save_to_tsv([[title_abstract, main_categories]])
    return redirect(url_for('index'))

@app.route('/upload_tsv', methods=['POST'])
def upload_tsv():
    file = request.files['file']
    if file and file.filename.endswith('.tsv'):
        try:
            # Read the file into a DataFrame
            data = pd.read_csv(file, sep='\t', header=0)  # Ensure header is read correctly
            
            # Check for consistent number of columns
            expected_columns = ['title', 'main_categories']
            if list(data.columns) != expected_columns:
                raise ValueError('Unexpected columns in the TSV file.')

            # Check if the DataFrame is empty
            if data.empty:
                print('The uploaded file is empty. Please upload a valid TSV file.')
            else:
                save_to_tsv(data.values.tolist())
        except pd.errors.EmptyDataError:
            print('No data found in the uploaded file. Please upload a valid TSV file.')
        except ValueError as ve:
            print(f'ValueError: {ve}')
        except Exception as e:
            print(f'An error occurred: {e}')
    else:
        print('Invalid file format. Please upload a TSV file.')
    return redirect(url_for('index'))


@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['predict_input']
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Extract input ids, attention masks, and token type ids for the model
    input_ids = inputs['input_ids']
    attn_mask = inputs['attention_mask']
    token_type_ids = inputs.get('token_type_ids', torch.zeros_like(input_ids))  # Optional token type ids

    with torch.no_grad():
        # Get the model's raw outputs
        outputs = model(input_ids, attn_mask, token_type_ids)
    
    # Get the class with the highest probability
    predictions = torch.argmax(outputs, dim=1)
    
    label_map = {
        0: 'astro-ph',
        1: 'cs',
        2: 'hep-ph',
        3: 'hep-th',
        4: 'math',
        5: 'physics',
        6: 'quant-ph',
        7: 'combined'
    }

    # Map the prediction index to the label
    predicted_label = label_map.get(predictions.item(), 'Unknown Label')
    
    # Format the output for display
    output_string = f"{predicted_label}"

    # Render the prediction on the web page
    return render_template('index.html', prediction=output_string, input_text=input_text)

@app.route('/feedback', methods=['POST'])
def feedback():
    input_text = request.form.get('input_text', '')
    predicted_output = request.form.get('predicted_output', '')
    feedback_type = request.form.get('feedback_type', '')

    if feedback_type == 'correct':
        # Save the correct feedback
        save_to_tsv([[input_text, predicted_output]])
    elif feedback_type == 'incorrect':
        correct_output = request.form.get('correct_output', '')
        if correct_output:
            save_to_tsv([[input_text, correct_output]])
        else:
            print("Error: Correct output not provided for incorrect feedback.")
    
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
