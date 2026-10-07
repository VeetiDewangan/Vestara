import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../../../core/constants/app_colors.dart';
import '../data/clothing_analysis.dart';
import '../data/clothing_api.dart';
import '../data/clothing_taxonomy.dart';
import '../data/wardrobe_item.dart';
import '../data/wardrobe_store.dart';
import '../widgets/wardrobe_image.dart';

/// Pick a photo -> the backend removes the background and tags it ->
/// the user reviews category + tags -> the check button saves it to the wardrobe.
class AddItemScreen extends StatefulWidget {
  const AddItemScreen({super.key, this.api, this.store});

  /// Injectable for tests; default to the real backend client / app store.
  final ClothingApi? api;
  final WardrobeStore? store;

  @override
  State<AddItemScreen> createState() => _AddItemScreenState();
}

class _AddItemScreenState extends State<AddItemScreen> {
  final ImagePicker _picker = ImagePicker();
  final TextEditingController _tagController = TextEditingController();
  late final ClothingApi _api = widget.api ?? ClothingApi();
  late final WardrobeStore _store = widget.store ?? WardrobeStore.instance;

  // Original photo (shown while the cut-out is being made / if it fails).
  Uint8List? _pickedBytes;
  String _pickedName = 'upload.jpg';

  // Result from the backend.
  ClothingAnalysis? _result;
  bool _isAnalyzing = false;
  String? _error;
  bool _isSaving = false;
  bool _isRetagging = false;
  List<String> _tags = [];

  /// Tags the user can add with one tap (used most when AI tagging is off).
  static const List<String> _quickTags = [
    // Occasions
    'daily wear', 'office', 'date night', 'wedding', 'party', 'festive',
    'college', 'travel', 'gym', 'beach',
    // Style
    'casual', 'formal', 'ethnic', 'sporty', 'elegant',
    // Season
    'summer', 'winter', 'monsoon', 'all-season',
    // Pattern
    'solid', 'floral', 'striped', 'printed', 'embroidered',
    // Fit
    'slim fit', 'oversized', 'a-line', 'bodycon',
    // Material
    'cotton', 'denim', 'silk', 'chiffon', 'linen',
  ];

  String selectedCategory = 'Top';
  String selectedSubcategory = 'T-Shirt';

  @override
  void dispose() {
    _tagController.dispose();
    if (widget.api == null) _api.dispose();
    super.dispose();
  }

  Future<void> _pickImage(ImageSource source) async {
    XFile? image;
    try {
      image = await _picker.pickImage(
        source: source,
        maxWidth: 2048, // backend downsizes anyway; this speeds up the upload
        imageQuality: 90,
      );
    } catch (_) {
      _showMessage("Couldn't open the ${source == ImageSource.camera ? 'camera' : 'gallery'}.");
      return;
    }
    if (image == null) return; // user cancelled

    final name = image.name;
    final bytes = await image.readAsBytes();
    if (!mounted) return;
    setState(() {
      _pickedBytes = bytes;
      _pickedName = name;
      _result = null;
      _error = null;
      _tags = [];
    });
    await _analyze();
  }

  Future<void> _analyze() async {
    final bytes = _pickedBytes;
    if (bytes == null) return;
    setState(() {
      _isAnalyzing = true;
      _error = null;
    });
    try {
      final result = await _api.analyzeBytes(bytes, _pickedName);
      if (!mounted) return;
      final match = ClothingTaxonomy.fromAiCategory(result.tags.category);
      setState(() {
        _result = result;
        _tags = List<String>.from(result.tagList);
        if (match != null) {
          selectedCategory = match.group;
          selectedSubcategory = match.sub;
        }
        _isAnalyzing = false;
      });
    } on ClothingApiException catch (e) {
      _fail(e.message);
    } catch (_) {
      _fail('Something went wrong. Please try again.');
    }
  }

  /// Asks the backend to run only the AI tagging step again.
  Future<void> _retag() async {
    final current = _result;
    if (current == null || _isRetagging || _isAnalyzing) return;
    setState(() => _isRetagging = true);
    try {
      final updated = await _api.retag(current);
      if (!mounted) return;
      final match = ClothingTaxonomy.fromAiCategory(updated.tags.category);
      setState(() {
        _result = updated;
        // keep what the user already typed/kept, add the new AI tags
        for (final tag in updated.tagList) {
          if (!_tags.contains(tag)) _tags.add(tag);
        }
        if (match != null) {
          selectedCategory = match.group;
          selectedSubcategory = match.sub;
        }
        _isRetagging = false;
      });
      if (!updated.tagging.isAi) {
        _showMessage(_tagProblem(updated.tagging.error));
      }
    } on ClothingApiException catch (e) {
      if (!mounted) return;
      setState(() => _isRetagging = false);
      _showMessage(e.message);
    } catch (_) {
      if (!mounted) return;
      setState(() => _isRetagging = false);
      _showMessage('Something went wrong. Please try again.');
    }
  }

  /// Turns the backend's technical reason into something actionable.
  static String _tagProblem(String? error) {
    final e = (error ?? '').toLowerCase();
    if (e.contains('gemini_api_key is not set')) {
      return 'No Gemini key is set on the server. Add GEMINI_API_KEY to '
          'backend/.env and restart the backend.';
    }
    if (e.contains('api key')) {
      return 'Google rejected the API key. Check GEMINI_API_KEY in backend/.env.';
    }
    if (e.contains('rate limit') || e.contains('quota')) {
      return "Google's free tagging quota is used up for the moment. Try again in a minute.";
    }
    if (e.contains('overloaded') || e.contains('503') || e.contains('unavailable')) {
      return "Google's AI is busy right now (this is temporary, not your setup). "
          'Wait a few seconds and tap "Try smart tags again".';
    }
    if (e.contains('not found')) {
      return 'None of the configured AI models exist any more. Set GEMINI_MODEL '
          'in backend/.env to a current model from Google AI Studio.';
    }
    if (e.contains('network')) {
      return "The server couldn't reach Google. Check its internet connection.";
    }
    if (e.isEmpty) return 'AI tagging did not run.';
    return error!;
  }

  void _fail(String message) {
    if (!mounted) return;
    setState(() {
      _error = message;
      _isAnalyzing = false;
    });
  }

  void _showMessage(String text) {
    if (!mounted) return;
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(SnackBar(content: Text(text)));
  }

  void _addTag(String raw) {
    final tag = raw.trim().toLowerCase();
    _tagController.clear();
    if (tag.isEmpty || _tags.contains(tag)) return;
    setState(() => _tags.add(tag));
  }

  Future<void> _save() async {
    if (_isSaving) return;
    if (_pickedBytes == null) {
      _showMessage('Please add a photo first.');
      return;
    }
    if (_isAnalyzing) {
      _showMessage('Still processing your photo…');
      return;
    }
    final result = _result;
    if (result == null) {
      _showMessage('The photo has not been processed yet. Tap Retry.');
      return;
    }
    setState(() => _isSaving = true);
    await _store.add(WardrobeItem(
      id: DateTime.now().microsecondsSinceEpoch.toString(),
      filename: result.filename,
      category: selectedCategory,
      subcategory: selectedSubcategory,
      tags: List<String>.from(_tags),
      colors: result.tags.colors,
      createdAt: DateTime.now(),
      imageData: result.imageBase64,
    ));
    if (!mounted) return;
    Navigator.pop(context, true);
  }

  void _showImageSourceSheet() {
    if (_isAnalyzing) return;
    showModalBottomSheet(
      context: context,
      backgroundColor: AppColors.cardBg,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) => Container(
        padding: const EdgeInsets.all(20),
        child: Wrap(
          children: [
            ListTile(
              leading: const Icon(Icons.camera_alt, color: Color.fromARGB(255, 62, 4, 3)),
              title: const Text('Take Photo', style: TextStyle(color: AppColors.textDark, fontWeight: FontWeight.bold)),
              onTap: () {
                Navigator.pop(context);
                _pickImage(ImageSource.camera);
              },
            ),
            ListTile(
              leading: const Icon(Icons.photo_library, color: Color.fromARGB(255, 62, 4, 3)),
              title: const Text('Choose from Gallery', style: TextStyle(color: AppColors.textDark, fontWeight: FontWeight.bold)),
              onTap: () {
                Navigator.pop(context);
                _pickImage(ImageSource.gallery);
              },
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final subcategories = ClothingTaxonomy.subcategories[selectedCategory] ?? const <String>[];

    return Scaffold(
      appBar: AppBar(
        title: const Text('Add Item'),
        actions: [
          IconButton(
            icon: const Icon(Icons.check, size: 26),
            onPressed: _save,
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            GestureDetector(
              onTap: _showImageSourceSheet,
              child: Container(
                height: 240,
                width: double.infinity,
                decoration: BoxDecoration(
                  color: AppColors.cardBg,
                  borderRadius: BorderRadius.circular(24),
                  border: Border.all(color: AppColors.primaryWine.withOpacity(0.3), width: 2),
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(22),
                  child: _buildPhotoArea(),
                ),
              ),
            ),
            if (_error != null) ...[
              const SizedBox(height: 12),
              Row(
                children: [
                  const Icon(Icons.error_outline, color: AppColors.primaryWine, size: 20),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(_error!, style: const TextStyle(color: AppColors.textDark, fontSize: 13)),
                  ),
                  TextButton(
                    onPressed: _isAnalyzing ? null : _analyze,
                    child: const Text('Retry', style: TextStyle(color: AppColors.primaryWine, fontWeight: FontWeight.bold)),
                  ),
                ],
              ),
            ],
            const SizedBox(height: 24),
            const Text('Select Category', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: AppColors.textDark)),
            const SizedBox(height: 12),
            Row(
              children: [
                for (var i = 0; i < ClothingTaxonomy.groups.length; i++) ...[
                  if (i > 0) const SizedBox(width: 8),
                  _buildCategoryChip(ClothingTaxonomy.groups[i]),
                ],
              ],
            ),
            const SizedBox(height: 24),
            const Text('Select Subcategory', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: AppColors.textDark)),
            const SizedBox(height: 12),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                for (final sub in subcategories) _buildSubcategoryChip(sub),
              ],
            ),
            if (_result != null) ..._buildTagsSection(_result!),
          ],
        ),
      ),
    );
  }

  /// Placeholder -> original photo (while processing) -> transparent cut-out.
  Widget _buildPhotoArea() {
    final result = _result;
    Widget content;
    if (result != null) {
      content = Container(
        color: AppColors.cardBg,
        padding: const EdgeInsets.all(12),
        child: WardrobeImage(
          bytes: result.imageBytes,
          url: result.imageUrl,
          fit: BoxFit.contain,
          iconSize: 40,
        ),
      );
    } else if (_pickedBytes != null) {
      content = Image.memory(_pickedBytes!, fit: BoxFit.cover);
    } else {
      return Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: const [
          Icon(Icons.add_a_photo_outlined, size: 50, color: AppColors.primaryWine),
          SizedBox(height: 12),
          Text(
            'Tap to add photo of your clothing item',
            style: TextStyle(color: AppColors.textDark, fontWeight: FontWeight.w600),
          ),
        ],
      );
    }

    return Stack(
      fit: StackFit.expand,
      children: [
        content,
        if (_isAnalyzing)
          Container(
            color: Colors.black38,
            child: const Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                CircularProgressIndicator(color: Colors.white),
                SizedBox(height: 12),
                Text(
                  'Removing background…',
                  style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
                ),
                SizedBox(height: 4),
                Text(
                  'The first photo can take a little longer',
                  style: TextStyle(color: Colors.white70, fontSize: 12),
                ),
              ],
            ),
          ),
      ],
    );
  }

  List<Widget> _buildTagsSection(ClothingAnalysis result) {
    return [
      const SizedBox(height: 24),
      const Text('Tags', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: AppColors.textDark)),
      const SizedBox(height: 12),
      if (!result.tagging.isAi)
        Container(
          margin: const EdgeInsets.only(bottom: 12),
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: AppColors.cardBg,
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: AppColors.cardBorder),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Icon(Icons.info_outline, color: AppColors.primaryWine, size: 20),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      'Smart tagging is off, so only basic tags were found. '
                      '${_tagProblem(result.tagging.error)}',
                      style: const TextStyle(color: AppColors.textDark, fontSize: 13),
                    ),
                  ),
                ],
              ),
              Align(
                alignment: Alignment.centerRight,
                child: TextButton.icon(
                  onPressed: _isRetagging ? null : _retag,
                  icon: _isRetagging
                      ? const SizedBox(
                          width: 16,
                          height: 16,
                          child: CircularProgressIndicator(strokeWidth: 2, color: AppColors.primaryWine),
                        )
                      : const Icon(Icons.auto_awesome, size: 18, color: AppColors.primaryWine),
                  label: const Text(
                    'Try smart tags again',
                    style: TextStyle(color: AppColors.primaryWine, fontWeight: FontWeight.bold),
                  ),
                ),
              ),
            ],
          ),
        ),
      Wrap(
        spacing: 8,
        runSpacing: 4,
        children: [
          for (final tag in _tags)
            InputChip(
              label: Text(tag, style: const TextStyle(color: AppColors.textDark, fontWeight: FontWeight.w600)),
              backgroundColor: AppColors.cardBg,
              side: const BorderSide(color: AppColors.cardBorder),
              deleteIconColor: AppColors.primaryWine,
              onDeleted: () => setState(() => _tags.remove(tag)),
            ),
        ],
      ),
      const SizedBox(height: 8),
      TextField(
        controller: _tagController,
        textInputAction: TextInputAction.done,
        style: const TextStyle(color: AppColors.textDark),
        decoration: InputDecoration(
          hintText: 'Add a tag (e.g. linen)',
          hintStyle: const TextStyle(color: AppColors.textMuted),
          filled: true,
          fillColor: AppColors.cardBg,
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: const BorderSide(color: AppColors.cardBorder),
          ),
          suffixIcon: IconButton(
            icon: const Icon(Icons.add, color: AppColors.primaryWine),
            onPressed: () => _addTag(_tagController.text),
          ),
        ),
        onSubmitted: _addTag,
      ),
      const SizedBox(height: 12),
      const Text('Quick add', style: TextStyle(fontSize: 13, color: AppColors.textMuted)),
      const SizedBox(height: 6),
      Wrap(
        spacing: 8,
        runSpacing: 0,
        children: [
          for (final tag in _quickTags)
            if (!_tags.contains(tag))
              ActionChip(
                label: Text(tag, style: const TextStyle(color: AppColors.textDark, fontSize: 12)),
                backgroundColor: AppColors.background,
                side: const BorderSide(color: AppColors.cardBorder),
                onPressed: () => _addTag(tag),
              ),
        ],
      ),
      const SizedBox(height: 24),
    ];
  }

  Widget _buildCategoryChip(String label) {
    final isSelected = selectedCategory == label;
    return Expanded(
      child: GestureDetector(
        onTap: () => setState(() {
          selectedCategory = label;
          final subs = ClothingTaxonomy.subcategories[label] ?? const <String>[];
          if (!subs.contains(selectedSubcategory) && subs.isNotEmpty) {
            selectedSubcategory = subs.first;
          }
        }),
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 12),
          decoration: BoxDecoration(
            color: isSelected ? Color.fromARGB(255, 62, 4, 3) : AppColors.cardBg,
            borderRadius: BorderRadius.circular(12),
          ),
          alignment: Alignment.center,
          child: FittedBox(
            fit: BoxFit.scaleDown,
            child: Text(
              label,
              style: TextStyle(
                color: isSelected ? AppColors.textLight : AppColors.textDark,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildSubcategoryChip(String label) {
    final isSelected = selectedSubcategory == label;
    return GestureDetector(
      onTap: () => setState(() => selectedSubcategory = label),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
        decoration: BoxDecoration(
          color: isSelected ? Color.fromARGB(255, 62, 4, 3) : AppColors.cardBg,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Text(
          label,
          style: TextStyle(
            color: isSelected ? AppColors.textLight : AppColors.textDark,
            fontWeight: FontWeight.bold,
          ),
        ),
      ),
    );
  }
}
