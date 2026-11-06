
def assign_priority(impact, urgency):

    impact = impact.lower()
    urgency = urgency.lower()

    if impact == "high" and urgency == "high":
        return "P1"

    elif impact == "high" or urgency == "high":
        return "P2"

    elif impact == "medium" and urgency == "medium":
        return "P3"

    else:
        return "P4"


if __name__ == "__main__":

    test_cases = [
        ("high", "high"),
        ("high", "low"),
        ("medium", "medium"),
        ("low", "low")
    ]

    for impact, urgency in test_cases:

        priority = assign_priority(impact, urgency)

        print(
            f"Impact: {impact}, "
            f"Urgency: {urgency}, "
            f"Priority: {priority}"
        )
