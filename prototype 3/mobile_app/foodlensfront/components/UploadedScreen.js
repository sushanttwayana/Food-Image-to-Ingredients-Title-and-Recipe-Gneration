import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Image,
  SafeAreaView,
  Alert,
  ScrollView,
} from 'react-native';
import { launchImageLibrary } from 'react-native-image-picker';
import Ionicons from 'react-native-vector-icons/Ionicons';

const UploadedScreen = () => {
  const [selectedImage, setSelectedImage] = useState(null);

  const handleImageUpload = () => {
    const options = {
      mediaType: 'photo',
      includeBase64: false,
      maxHeight: 2000,
      maxWidth: 2000,
    };

    launchImageLibrary(options, (response) => {
      if (response.didCancel || response.error) {
        console.log('User cancelled or error');
      } else if (response.assets && response.assets[0]) {
        setSelectedImage(response.assets[0].uri);
      }
    });
  };

  const handleViewRecipe = () => {
    if (!selectedImage) {
      Alert.alert('No Image', 'Please upload an image first');
      return;
    }
    // Navigate to recipe view or process image
    console.log('View recipe for image:', selectedImage);
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView style={styles.content}>
        {/* Header */}
        <View style={styles.header}>
          <View style={styles.logoContainer}>
            <Image source={require('../assets/logo.png')} style={styles.logo} />
          </View>
        </View>

        {/* Title Section */}
        <View style={styles.titleSection}>
          <View style={styles.iconContainer}>
            <View style={styles.plateIcon}>
              <View style={styles.plateInner} />
            </View>
          </View>
          <Text style={styles.headerText}> Image Uploaded</Text>
        </View>

        {/* Upload Area */}
        <TouchableOpacity 
          style={styles.uploadArea} 
          onPress={handleImageUpload}
          activeOpacity={0.7}
        >
          {selectedImage ? (
            <Image source={{ uri: selectedImage }} style={styles.uploadedImage} />
          ) : (
            <View style={styles.placeholderContent}>
              {/* Background Image with Low Opacity */}
              <Image 
                source={require('../assets/newafood.png')} 
                style={styles.backgroundImage}
              />
              <View style={styles.overlayPattern}>
                {/* Overlay pattern for additional visual effect */}
                <View style={styles.patternOverlay} />
              </View>
            </View>
          )}
        </TouchableOpacity>

        {/* Upload Button */}
        <TouchableOpacity 
          style={styles.uploadButton} 
          onPress={handleImageUpload}
          activeOpacity={0.8}
        >
          <Text style={styles.uploadButtonIcon}>⬆</Text>
          <Text style={styles.uploadButtonText}>Upload your Image Here</Text>
        </TouchableOpacity>

        {/* View Recipe Button */}
        <TouchableOpacity 
          style={styles.recipeButton} 
          onPress={handleViewRecipe}
          activeOpacity={0.8}
        >
          <Text style={styles.recipeButtonText}>View Recipe & Ingredients</Text>
          <Text style={styles.recipeButtonArrow}>→</Text>
        </TouchableOpacity>
      </ScrollView>

      {/* Bottom Navigation */}
      <View style={styles.bottomNav}>
        <TouchableOpacity style={styles.navItem}>
          <Ionicons name="home" size={24} color="#999" />
        </TouchableOpacity>
        <TouchableOpacity style={[styles.navItem, styles.activeNavItem]}>
          <Ionicons name="restaurant" size={24} color="#D85A47" />
        </TouchableOpacity>
        <TouchableOpacity style={styles.navItem}>
          <Ionicons name="search" size={24} color="#999" />
        </TouchableOpacity>
        <TouchableOpacity style={styles.navItem}>
          <Ionicons name="person" size={24} color="#999" />
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  content: {
    flex: 1,
    paddingHorizontal: 20,
    paddingTop: 60,
  },
  header: {
    alignItems: 'center',
    marginBottom: 20,
  },
  logoContainer: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  logo: {
    width: 80,
    height: 80,
    resizeMode: 'contain',
  },
  titleSection: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 30,
    marginTop:30
  },
  iconContainer: {
    marginRight: 15,
  },
  plateIcon: {
    width: 30,
    height: 30,
    borderRadius: 15,
    borderWidth: 2,
    borderColor: '#D85A47',
    justifyContent: 'center',
    alignItems: 'center',
  },
  plateInner: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: '#D85A47',
  },
  headerText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  uploadArea: {
    height: 200,
    borderRadius: 15,
    backgroundColor: '#E8E5E0',
    marginBottom: 30,
    overflow: 'hidden',
    position: 'relative',
  },
  uploadedImage: {
    width: '100%',
    height: '100%',
    borderRadius: 15,
  },
  placeholderContent: {
    flex: 1,
    position: 'relative',
    justifyContent: 'center',
    alignItems: 'center',
  },
  backgroundImage: {
    position: 'absolute',
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
    // opacity: 0.20,
  },
  overlayPattern: {
    position: 'absolute',
    width: '100%',
    height: '100%',
    backgroundColor: 'rgba(248, 245, 240, 0.3)',
  },
  patternOverlay: {
    flex: 1,
    backgroundColor: 'transparent',
  },
  uploadButton: {
    backgroundColor: '#D85A47',
    borderRadius: 12,
    paddingVertical: 16,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 20,
  },
  uploadButtonIcon: {
    color: 'white',
    fontSize: 18,
    marginRight: 10,
    fontWeight: 'bold',
  },
  uploadButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  recipeButton: {
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderColor: '#999',
    borderRadius: 12,
    paddingVertical: 16,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  recipeButtonText: {
    color: '#666',
    fontSize: 16,
    fontWeight: '500',
    marginRight: 10,
  },
  recipeButtonArrow: {
    color: '#666',
    fontSize: 16,
  },
  bottomNav: {
    flexDirection: 'row',
    backgroundColor: '#2A2A2A',
    paddingVertical: 15,
    paddingHorizontal: 20,
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
  },
  navItem: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: 10,
  },
  activeNavItem: {
    backgroundColor: 'rgba(216, 90, 71, 0.15)',
    borderRadius: 8,
  },

});

export default UploadedScreen;