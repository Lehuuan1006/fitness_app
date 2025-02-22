import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:http/http.dart' as http;
import 'package:provider/provider.dart';

void main() async {
  await dotenv.load(fileName: ".env");
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (context) => ChatProvider(),
      child: MaterialApp(
        title: 'Fitness AI Assistant',
        theme: ThemeData(primarySwatch: Colors.blue),
        home: ChatScreen(),
      ),
    );
  }
}

class ChatScreen extends StatelessWidget {
  final TextEditingController _controller = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Fitness AI Assistant")),
      body: Column(
        children: [
          Expanded(child: Consumer<ChatProvider>(
            builder: (context, chatProvider, child) {
              return ListView.builder(
                itemCount: chatProvider.messages.length,
                itemBuilder: (context, index) {
                  final message = chatProvider.messages[index];
                  return ListTile(
                    title: Align(
                      alignment: message.isUser ? Alignment.centerRight : Alignment.centerLeft,
                      child: Container(
                        padding: EdgeInsets.all(10),
                        color: message.isUser ? Colors.blue[100] : Colors.grey[300],
                        child: Text(message.text),
                      ),
                    ),
                  );
                },
              );
            },
          )),
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _controller,
                    decoration: InputDecoration(hintText: 'Type your question...'),
                  ),
                ),
                IconButton(
                  icon: Icon(Icons.send),
                  onPressed: () {
                    if (_controller.text.isNotEmpty) {
                      context.read<ChatProvider>().sendMessage(_controller.text);
                      _controller.clear();
                    }
                  },
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class ChatProvider with ChangeNotifier {
  List<Message> messages = [];

  Future<void> sendMessage(String text) async {
    messages.add(Message(text: text, isUser: true));
    notifyListeners();

    final response = await queryPinecone(text);
    messages.add(Message(text: response, isUser: false));
    notifyListeners();
  }

  Future<String> queryPinecone(String query) async {
    final apiUrl = 'https://your-api-endpoint'; // Thay đổi thành endpoint của bạn
    final response = await http.post(
      Uri.parse(apiUrl),
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ${dotenv.env['PINECONE_API_KEY']}',
      },
      body: jsonEncode({'query': query}),
    );

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return data['response'] ?? 'No response from server.';
    } else {
      return 'Error: ${response.reasonPhrase}';
    }
  }
}

class Message {
  final String text;
  final bool isUser;

  Message({required this.text, required this.isUser});
}
