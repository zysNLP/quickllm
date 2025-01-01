def verify_identity(booking_reference: str, full_name: str, flight_number: str) -> bool:
    """
    Verifies the customer's identity using booking reference, full name, and flight number.
    
    Args:
        booking_reference: Customer's booking reference.
        full_name: Customer's full name.
        flight_number: Flight number.
    
    Returns:
        bool: True if identity is verified, False otherwise.
    """
    # 模拟数据库中的预订信息
    mock_bookings = {
        "ABC123": {
            "name": "John Smith",
            "flight": "CA890"
        },
        "XYZ789": {
            "name": "Jane Doe",
            "flight": "MU456"
        }
    }
    
    # 验证逻辑
    if booking_reference in mock_bookings:
        booking = mock_bookings[booking_reference]
        return (
            booking["name"].lower() == full_name.lower() and 
            booking["flight"].lower() == flight_number.lower()
        )
    return False

def ask_clarification(prompt: str) -> str:
    """
    Prompts the customer for clarification on their request.
    
    Args:
        prompt: The prompt to ask the customer.
    
    Returns:
        str: Customer's response with some basic validation
    """
    # 模拟一些常见的问题和答案
    clarification_responses = {
        "baggage allowance": "Your baggage allowance is 23kg for checked baggage.",
        "meal preference": "We offer regular, vegetarian, and halal meal options.",
        "check-in time": "Check-in opens 3 hours before departure.",
        "flight status": "Please provide your flight number for status updates."
    }
    
    # 简单的关键词匹配
    prompt_lower = prompt.lower()
    for key in clarification_responses:
        if key in prompt_lower:
            return clarification_responses[key]
    
    return "I'm sorry, could you please be more specific about your question?" 


if __name__ == "__main__":
        # 测试身份验证
    print(verify_identity("ABC123", "John Smith", "CA890"))  # True
    print(verify_identity("ABC123", "Wrong Name", "CA890"))  # False

    # 测试澄清问题
    print(ask_clarification("What is the baggage allowance?"))  # 返回行李限额信息
    print(ask_clarification("What meal options do you have?"))  # 返回餐食选项信息