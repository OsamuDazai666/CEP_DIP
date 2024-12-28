import csv
import logging

# Configure logging
logging.basicConfig(
    filename='model_accuracy.log', 
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def calculate_accuracy(csv_file_path):
    try:
        # Initialize counters
        total_entries = 0
        correct_predictions = 0

        # Read the CSV file
        with open(csv_file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                total_entries += 1
                if row['Truth'] == row['Prediction']:
                    correct_predictions += 1

        # Calculate accuracy
        if total_entries == 0:
            accuracy = 0.0
        else:
            accuracy = (correct_predictions / total_entries) * 100

        # Log the results
        logging.info(f'Total entries: {total_entries}')
        logging.info(f'Correct predictions: {correct_predictions}')
        logging.info(f'Model accuracy: {accuracy:.2f}%')

        print("Logs generated successfully. Check 'model_accuracy.log' for details.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print("An error occurred. Check 'model_accuracy.log' for details.")

# Replace 'data.csv' with the path to your CSV file
calculate_accuracy('predictions/effnet_pred.csv')
