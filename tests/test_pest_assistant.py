from app.pest_assistant import load_class_names


def test_load_class_names_returns_expected_labels():
    labels = load_class_names()
    assert labels[0] == "grub"
    assert len(labels) == 10

