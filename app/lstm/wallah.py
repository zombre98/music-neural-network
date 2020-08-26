train = reduced_dataset[reduced_dataset.split == "train"]
test = reduced_dataset[reduced_dataset.split == "test"]
validation = reduced_dataset[reduced_dataset.split == "validation"]
# x_train = np.vstack(train["padded_notes"]).reshape(len(train.index), -1, features)
x_train = np.vstack(train["trimed_notes"]).reshape(len(train.index), -1, features)
y_train = np.asarray(train["canonical_composer"].cat.codes)
# x_test = np.vstack(test["padded_notes"]).reshape(len(test.index), -1, features)
x_test = np.vstack(test["trimed_notes"]).reshape(len(test.index), -1, features)
y_test = np.asarray(test["canonical_composer"].cat.codes)
# x_validation = np.vstack(validation["padded_notes"]).reshape(len(validation.index), -1, features)
x_validation = np.vstack(validation["trimed_notes"]).reshape(len(validation.index), -1, features)
y_validation = np.asarray(validation["canonical_composer"].cat.codes)

print("dataset created:")
print(
    f"{x_train.shape[0]*100/(x_train.shape[0]+x_test.shape[0]):.0f}% of the data for training"
)
print(f"train: {x_train.shape}, {y_train.shape}")
print(f"test:  {x_test.shape}, {y_test.shape}")

test_model = keras.models.load_model(os.path.join(os.getcwd(), "saved_models", "the_one.hdf5"))

idx = np.random.choice(len(x_validation))
sample, sample_label = x_validation[idx], y_validation[idx]
result = tf.argmax(test_model.predict_on_batch(tf.expand_dims(sample, 0)), axis=1)
print(
    f"Predicted result is: {code_to_label(result.numpy())}, target result is: {code_to_label([sample_label])}"
)