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
  ActivityIndicator,
} from 'react-native';
import { launchImageLibrary } from 'react-native-image-picker';
import Ionicons from 'react-native-vector-icons/Ionicons';

const ImageUploadScreen1 = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [isUploading, setIsUploading] = useState(false);

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
        setIsUploading(true);
        // Simulate upload process
        setTimeout(() => {
          setSelectedImage(response.assets[0].uri);
          setIsUploading(false);
        }, 2000);
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

  const handleClose = () => {
    // Handle close/back navigation
    console.log('Close pressed');
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView style={styles.content}>
        {/* Logo Header */}
        <View style={styles.logoHeader}>
          <View style={styles.logoContainer}>
            <Image source={require('../assets/logo.png')} style={styles.logo} />
          </View>
        </View>

        {/* Header */}
        <View style={styles.header}>
          <View style={styles.headerLeft}>
            <View style={styles.plateIcon}>
              <View style={styles.plateInner} />
            </View>
            <Text style={styles.headerText}>Image Uploading</Text>
          </View>
          <TouchableOpacity style={styles.closeButton} onPress={handleClose}>
            <Ionicons name="close" size={24} color="#D85A47" />
          </TouchableOpacity>
        </View>

        {/* Upload Area */}
        <TouchableOpacity 
          style={styles.uploadArea} 
          onPress={handleImageUpload}
          activeOpacity={0.7}
          disabled={isUploading}
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
                {isUploading ? (
                  <ActivityIndicator size="large" color="#D85A47" />
                ) : (
                  <View style={styles.uploadIconContainer}>
                    <Ionicons name="scan-outline" size={48} color="#D85A47" />
                  </View>
                )}
              </View>
            </View>
          )}
        </TouchableOpacity>

        {/* Upload Button */}
        <TouchableOpacity 
          style={[styles.uploadButton, isUploading && styles.uploadButtonDisabled]} 
          onPress={handleImageUpload}
          activeOpacity={0.8}
          disabled={isUploading}
        >
          <Ionicons name="cloud-upload" size={20} color="white" style={styles.uploadButtonIcon} />
          <Text style={styles.uploadButtonText}>
            {isUploading ? 'Uploading...' : 'Upload your Image Here'}
          </Text>
        </TouchableOpacity>

        {/* View Recipe Button */}
        <TouchableOpacity 
          style={styles.recipeButton} 
          onPress={handleViewRecipe}
          activeOpacity={0.8}
        >
          <Text style={styles.recipeButtonText}>View Recipe & Ingredients</Text>
          <Ionicons name="arrow-forward" size={16} color="#666" style={styles.recipeButtonArrow} />
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
        {/* <TouchableOpacity style={styles.navItem}>
          <Ionicons name="search" size={24} color="#999" />
        </TouchableOpacity> */}
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
    paddingTop: 20,
  },
  logoHeader: {
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
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 30,
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  plateIcon: {
    width: 30,
    height: 30,
    borderRadius: 15,
    borderWidth: 2,
    borderColor: '#D85A47',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
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
  closeButton: {
    width: 32,
    height: 32,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#D85A47',
    justifyContent: 'center',
    alignItems: 'center',
  },
  uploadArea: {
    height: 220,
    borderRadius: 15,
    backgroundColor: '#E8E5E0',
    marginBottom: 20,
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
    opacity: 0.15,
  },
  overlayPattern: {
    position: 'absolute',
    width: '100%',
    height: '100%',
    backgroundColor: 'rgba(248, 245, 240, 0.3)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  uploadIconContainer: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  uploadButton: {
    backgroundColor: '#999',
    borderRadius: 12,
    paddingVertical: 16,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 15,
  },
  uploadButtonDisabled: {
    opacity: 0.6,
  },
  uploadButtonIcon: {
    marginRight: 10,
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
    marginRight: 8,
  },
  recipeButtonArrow: {
    marginLeft: 4,
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
export default ImageUploadScreen1