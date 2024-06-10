

curl -X POST http://localhost:8000/signup -H "Content-Type: application/json" -d '{"username": "newuser", "password": "newpassword", "email": "newuser@example.com"}'

curl -X POST http://localhost:8000/login -H "Content-Type: application/json" -d '{"username": "newuser", "password": "newpassword"}'


# Use the token obtained from the login/signup response to send a message

curl -X GET http://localhost:8000/new -H "Content-Type: application/json" -H "Authorization: Token 79bf43ef85e150c8314731fce4278fd59fb788b5"

curl -X POST http://localhost:8000/send -H "Content-Type: application/json" -H "Authorization: Token 79bf43ef85e150c8314731fce4278fd59fb788b5" -d '{"conversation_id": 2, "content": "Hello, this is a test message"}'
