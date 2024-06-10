from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.authtoken.models import Token
from rest_framework import status

from django.shortcuts import get_object_or_404

from .models import Conversation, Message, User
from .utils import get_conversation_context, get_conversation_chain, get_vectorstore
from .serializers import *
from .types import ChatHistory


@api_view(["POST"])
def test(request):
    return Response({"test": True})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def send_message(request):
    user = request.user
    conversation_id = request.data.get("conversation_id")
    content = request.data.get("content")

    if not conversation_id or not content:
        return Response({"error": "Invalid data"}, status=400)

    try:
        conversation = Conversation.objects.get(id=conversation_id)
    except Conversation.DoesNotExist:
        return Response({"error": "Conversation not found"}, status=404)

    # Save the new message
    new_message = Message.objects.create(
        conversation=conversation, sender=user, content=content
    )

    # Get the context of the conversation
    messages = get_conversation_context(conversation_id)
    chat_history: list[ChatHistory] = [
        {"sender": msg.sender is not None, "content": msg.content} for msg in messages
    ]

    # Initialize conversation chain with context
    vectorstore = get_vectorstore()
    Lom7ami = get_conversation_chain(vectorstore, chat_history)

    # Generate response from ChatGPT
    user_message = {"question": content}
    response = Lom7ami(user_message)

    print("response", response)
    gpt_response = response["answer"].strip()

    # Save the GPT response as a new message
    bot_message = Message.objects.create(
        conversation=conversation, sender=None, content=gpt_response
    )

    return Response(
        {
            "user_message": MessageSerializer(new_message).data,
            "gpt_response": MessageSerializer(bot_message).data,
        }
    )


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def new_conversation(request):
    conversation = Conversation.objects.create(participant=request.user)
    conversation.save()
    return Response(ConversationSerializer(conversation).data)

@api_view(["POST"])
def signup(request):
    serializer = UserSerializerAuth(data=request.data)
    serializer_resp = UserSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        user = User.objects.get(username=request.data["username"])
        user.set_password(request.data["password"])
        user.save()
        token = Token.objects.create(user=user)
        return Response({"token": token.key, "user": serializer_resp.data})
    return Response(serializer.errors, status=status.HTTP_200_OK)


@api_view(["POST"])
def login(request):
    user = get_object_or_404(User, username=request.data["username"])
    if not user.check_password(request.data["password"]):
        return Response("missing user", status=status.HTTP_404_NOT_FOUND)
    token, created = Token.objects.get_or_create(user=user)
    serializer = UserSerializerAuth(user)
    return Response({"token": token.key, "user": serializer.data})
