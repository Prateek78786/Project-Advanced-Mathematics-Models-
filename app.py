import json
from flask import Flask, render_template, request, jsonify
from processing import analyze_upload

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 80 * 1024 * 1024
ALLOWED_EXTENSIONS = {'csv'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    uploaded_files = request.files.getlist('data_files')
    valid_files = [f for f in uploaded_files if f and allowed_file(f.filename)]

    if not valid_files:
        return jsonify({'success': False, 'message': 'Please upload one or more CSV files to analyze.'})

    result = analyze_upload(valid_files)
    if not result or result.get('errors'):
        message = 'Unable to analyze the uploaded dataset. '
        if result and result.get('errors'):
            message += ' '.join(result['errors'])
        return jsonify({'success': False, 'message': message})

    # Keep compatibility with React client by ensuring JSON serializable payload
    return jsonify({'success': True, 'result': result})


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'success': False, 'message': 'Uploaded file is too large. Limit is 80 MB.'}), 413


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8501)
