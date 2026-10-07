import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:image_picker/image_picker.dart';
import 'package:wardrobe_app/features/wardrobe/data/clothing_analysis.dart';
import 'package:wardrobe_app/features/wardrobe/data/clothing_api.dart';

XFile _fakePhoto() =>
    XFile.fromData(Uint8List.fromList([1, 2, 3]), name: 'shirt.jpg');

void main() {
  test('parses a full AI response and makes image_url absolute', () async {
    final client = MockClient((request) async {
      expect(request.method, 'POST');
      expect(request.url.path, '/api/v1/clothing/analyze');
      return http.Response(
        jsonEncode({
          'success': true,
          'filename': 'abc.png',
          'image_url': '/api/v1/clothing/result/abc.png',
          'width': 400,
          'height': 320,
          'processing_time_s': 3.1,
          'tags': {
            'category': 't-shirt',
            'colors': ['navy', 'white'],
            'pattern': 'striped',
            'style': ['casual'],
            'season': ['summer'],
            'material_guess': 'cotton',
            'extra_tags': ['crew neck'],
          },
          'tag_list': ['t-shirt', 'navy', 'white', 'striped'],
          'tagging': {'source': 'gemini', 'model': 'gemini-2.5-flash', 'error': null},
        }),
        200,
      );
    });

    final result = await ClothingApi(client: client).analyze(_fakePhoto());

    expect(result.imageUrl, endsWith('/api/v1/clothing/result/abc.png'));
    expect(result.imageUrl, startsWith('http'));
    expect(result.tags.category, 't-shirt');
    expect(result.tagList, contains('navy'));
    expect(result.tagging.isAi, isTrue);
  });

  test('handles the colors-only fallback (null tags) without crashing', () async {
    final client = MockClient((_) async => http.Response(
          jsonEncode({
            'filename': 'abc.png',
            'image_url': '/api/v1/clothing/result/abc.png',
            'width': 10,
            'height': 10,
            'processing_time_s': 1,
            'tags': {
              'category': null,
              'colors': ['white', 'beige'],
              'pattern': null,
              'style': [],
              'season': [],
              'material_guess': null,
              'extra_tags': [],
            },
            'tag_list': ['white', 'beige'],
            'tagging': {'source': 'local', 'model': null, 'error': 'no key'},
          }),
          200,
        ));

    final result = await ClothingApi(client: client).analyze(_fakePhoto());

    expect(result.tags.category, isNull);
    expect(result.tagList, ['white', 'beige']);
    expect(result.tagging.isAi, isFalse);
  });

  test('surfaces the backend error detail', () async {
    final client = MockClient((_) async => http.Response(
          jsonEncode({'detail': 'Image must be at least 256x256 pixels.'}),
          400,
        ));

    expect(
      () => ClothingApi(client: client).analyze(_fakePhoto()),
      throwsA(isA<ClothingApiException>().having(
          (e) => e.message, 'message', contains('256'))),
    );
  });

  test('keeps the picture that the backend embeds in the reply', () async {
    final pixels = base64Encode([137, 80, 78, 71, 1, 2, 3]);
    final client = MockClient((_) async => http.Response(
          jsonEncode({
            'filename': 'abc.png',
            'image_url': '/api/v1/clothing/result/abc.png',
            'image_base64': pixels,
            'image_mime': 'image/webp',
            'tags': {'colors': ['black']},
            'tag_list': ['black'],
            'tagging': {'source': 'local', 'error': 'x'},
          }),
          200,
        ));

    final result = await ClothingApi(client: client).analyze(_fakePhoto());

    expect(result.imageBase64, pixels);
    expect(result.imageBytes, [137, 80, 78, 71, 1, 2, 3]);
  });

  test('a broken embedded picture falls back to the image url', () async {
    final client = MockClient((_) async => http.Response(
          jsonEncode({
            'filename': 'abc.png',
            'image_url': '/api/v1/clothing/result/abc.png',
            'image_base64': '%%% not base64 %%%',
            'tags': {},
            'tag_list': [],
            'tagging': {'source': 'local'},
          }),
          200,
        ));

    final result = await ClothingApi(client: client).analyze(_fakePhoto());

    expect(result.imageBytes, isNull);
    expect(result.imageBase64, isNull);
    expect(result.imageUrl, endsWith('abc.png'));
  });

  test('retag only replaces the tags and keeps the picture', () async {
    final previous = ClothingAnalysis.fromJson({
      'filename': 'abc.png',
      'image_url': '/api/v1/clothing/result/abc.png',
      'image_base64': base64Encode([1, 2, 3]),
      'tags': {'colors': ['black']},
      'tag_list': ['black'],
      'tagging': {'source': 'local', 'error': 'rate limit'},
    }, baseUrl: 'http://host:8000');

    final client = MockClient((request) async {
      expect(request.method, 'POST');
      expect(request.url.path, '/api/v1/clothing/retag/abc.png');
      return http.Response(
        jsonEncode({
          'tags': {'category': 't-shirt', 'colors': ['black']},
          'tag_list': ['t-shirt', 'black', 'casual'],
          'tagging': {'source': 'gemini', 'model': 'gemini-3.5-flash', 'error': null},
        }),
        200,
      );
    });

    final updated = await ClothingApi(client: client).retag(previous);

    expect(updated.tagging.isAi, isTrue);
    expect(updated.tags.category, 't-shirt');
    expect(updated.tagList, ['t-shirt', 'black', 'casual']);
    expect(updated.imageBytes, [1, 2, 3]);
    expect(updated.filename, 'abc.png');
  });

  test('retag surfaces a missing cut-out as a readable error', () async {
    final previous = ClothingAnalysis.fromJson({
      'filename': 'gone.png',
      'tags': {},
      'tag_list': [],
      'tagging': {'source': 'local'},
    }, baseUrl: 'http://host:8000');
    final client = MockClient(
        (_) async => http.Response(jsonEncode({'detail': 'File not found.'}), 404));

    expect(
      () => ClothingApi(client: client).retag(previous),
      throwsA(isA<ClothingApiException>()
          .having((e) => e.message, 'message', contains('not found'))),
    );
  });
}
