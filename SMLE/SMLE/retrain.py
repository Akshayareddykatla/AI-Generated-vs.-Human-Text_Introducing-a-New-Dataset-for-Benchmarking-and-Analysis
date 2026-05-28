import json
from app import app, db, DetectionReport

def run_enhancement():
    """Label Enhancement: Promotes high-confidence submissions for model updates """
    with app.app_context():
        # Fetch reports not yet processed for retraining
        reports = DetectionReport.query.filter_by(is_processed=False).all()
        count = 0
        for r in reports:
            res = json.loads(r.label_results)
            # Threshold Check: Identify samples where the model is highly confident (>85%)
            if any(val > 0.85 for val in res.values()):
                r.is_processed = True
                count += 1
        db.session.commit()
        print(f"SSL Enhancement Complete: {count} samples promoted for retraining.")
def run_label_enhancement_loop(threshold=0.90):
    # Fetch reports that haven't been promoted for training yet
    unlabeled_reports = DetectionReport.query.filter_by(is_processed=False).all()
    
    for report in unlabeled_reports:
        # Load the raw model scores
        scores = json.loads(report.label_results)
        
        # Check if the model is confident enough to create an 'Enhanced Label'
        if any(score >= threshold for score in scores.values()):
            # Promotion logic: The model is now sure enough to treat this as ground truth
            report.is_processed = True 
            db.session.commit()
if __name__ == "__main__":
    run_enhancement()