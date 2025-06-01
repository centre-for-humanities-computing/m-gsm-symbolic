from pydantic_evals import Case

from m_gsm_symbolic.kaenguruen.load_data import KaenguruenProblem, load_kaenguruen


def test_load_kaenguruen():
    samples = load_kaenguruen()

    assert len(samples) == 106, "Expected 106 samples in the dataset"

    # test conversion to case
    for sample in samples:
        assert isinstance(sample, KaenguruenProblem), (
            "Sample should be of type KaenguruenProblem"
        )
        case = sample.to_case()
        assert isinstance(case, Case), "Converted case should be of type Case"

    ids = [s.to_case().name for s in samples]
    assert len(ids) == len(set(ids)), "Sample IDs should be unique"
