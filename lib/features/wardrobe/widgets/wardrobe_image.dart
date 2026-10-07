import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

import '../../../core/constants/app_colors.dart';

/// Shows a clothing cut-out.
///
/// * If [bytes] are given (the picture saved on the device) it just draws them.
/// * Otherwise it downloads [url] itself with `package:http` (the same way the
///   upload works), keeps the result in memory, and shows a spinner while it
///   loads and a tappable "retry" icon if it fails. `Image.network` was used
///   before and failed silently (blank card) on Flutter web.
class WardrobeImage extends StatefulWidget {
  const WardrobeImage({
    super.key,
    this.bytes,
    this.url,
    this.fit = BoxFit.contain,
    this.iconSize = 32,
  });

  final Uint8List? bytes;
  final String? url;
  final BoxFit fit;
  final double iconSize;

  /// Downloaded pictures, so scrolling / rebuilding never re-downloads them.
  static final Map<String, Uint8List> _memoryCache = {};

  @override
  State<WardrobeImage> createState() => _WardrobeImageState();
}

class _WardrobeImageState extends State<WardrobeImage> {
  Uint8List? _downloaded;
  bool _loading = false;
  bool _failed = false;

  @override
  void initState() {
    super.initState();
    _loadIfNeeded();
  }

  @override
  void didUpdateWidget(covariant WardrobeImage old) {
    super.didUpdateWidget(old);
    if (old.url != widget.url || old.bytes != widget.bytes) {
      _downloaded = null;
      _failed = false;
      _loadIfNeeded();
    }
  }

  /// [notify] is true only when called from a tap (outside initState /
  /// didUpdateWidget, where the framework rebuilds for us anyway).
  Future<void> _loadIfNeeded({bool notify = false}) async {
    final url = widget.url;
    if (widget.bytes != null || url == null || url.isEmpty) return;

    final cached = WardrobeImage._memoryCache[url];
    if (cached != null) {
      _downloaded = cached;
      return;
    }

    if (notify) {
      setState(() {
        _loading = true;
        _failed = false;
      });
    } else {
      _loading = true;
      _failed = false;
    }
    try {
      final response =
          await http.get(Uri.parse(url)).timeout(const Duration(seconds: 30));
      if (response.statusCode != 200 || response.bodyBytes.isEmpty) {
        throw http.ClientException('HTTP ${response.statusCode}');
      }
      WardrobeImage._memoryCache[url] = response.bodyBytes;
      if (!mounted || widget.url != url) return;
      setState(() {
        _downloaded = response.bodyBytes;
        _loading = false;
      });
    } catch (_) {
      if (!mounted || widget.url != url) return;
      setState(() {
        _loading = false;
        _failed = true;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final bytes = widget.bytes ?? _downloaded;
    if (bytes != null) {
      return Image.memory(
        bytes,
        fit: widget.fit,
        gaplessPlayback: true,
        errorBuilder: (context, error, stack) => _placeholder(Icons.broken_image_outlined),
      );
    }
    if (_loading) {
      return const Center(
        child: SizedBox(
          width: 22,
          height: 22,
          child: CircularProgressIndicator(strokeWidth: 2, color: AppColors.primaryWine),
        ),
      );
    }
    if (_failed) {
      return GestureDetector(
        onTap: () => _loadIfNeeded(notify: true),
        child: _placeholder(Icons.refresh, label: 'Tap to retry'),
      );
    }
    return _placeholder(Icons.checkroom);
  }

  Widget _placeholder(IconData icon, {String? label}) => Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: widget.iconSize, color: AppColors.primaryWine),
            if (label != null)
              Text(label, style: const TextStyle(fontSize: 10, color: AppColors.textMuted)),
          ],
        ),
      );
}
