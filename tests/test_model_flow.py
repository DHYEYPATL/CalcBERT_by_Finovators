from ml.adapter import predict_text, retrain_from_feedback

print("=== BEFORE RETRAIN ===")
print(predict_text("STARBCKS #1050 MUMBAI"))

print("\nApplying feedback update...")
print(retrain_from_feedback())

print("\n=== AFTER RETRAIN ===")
print(predict_text("STARBCKS #1050 MUMBAI"))
