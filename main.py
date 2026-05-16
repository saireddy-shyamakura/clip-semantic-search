import os
import logging
from features import extract_image_features
from index import Index
from store import Store
from search import image_search, text_search
from validation import (
    validate_image_path, validate_text_query, validate_folder_path,
    validate_choice, validate_positive_int, ValidationError
)
from add_images import add_images

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main application entry point with input validation."""
    try:
        # Initialize store and index
        store = Store()
        index = Index()
        
        # Load existing data
        store.load()
        
        # Add images from folder (batched)
        images_folder = os.path.join(os.path.dirname(__file__), "images")
        add_images(images_folder, store, index)
        
        # Build index if needed
        if store.count() > 0 and not index.is_built():
            all_features = [item["features"] for item in store.get_all()]
            index.build(all_features)

        logger.info(f"Database ready: {store.count()} images indexed")

        print("\n=== Image Search Engine ===")
        print("1. Image search (find similar images)")
        print("2. Text search (search by description)")
        
        while True:
            try:
                choice = input("\nChoice (1 or 2, or 'q' to quit): ").strip()
                
                # Validate choice
                is_valid, error_msg = validate_choice(choice, ["1", "2", "q"])
                if not is_valid:
                    print(f"Invalid input: {error_msg}")
                    continue
                
                if choice == "q":
                    print("Goodbye!")
                    break

                if choice == "1":
                    image_search_mode(index, store)

                elif choice == "2":
                    text_search_mode(index, store)

            except EOFError:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                print(f"Error: {e}")

    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        raise


def image_search_mode(index: Index, store: Store):
    """Interactive image search mode."""
    if store.count() == 0:
        print("No images in database. Add images first.")
        return
    
    print("\n--- Image Search Mode ---")
    while True:
        try:
            query_path = input("Image path (or 'back' to return): ").strip()
            
            if query_path.lower() == "back":
                break
            
            # Validate image path
            is_valid, error_msg = validate_image_path(query_path)
            if not is_valid:
                print(f"Invalid image: {error_msg}")
                continue
            
            # Validate top_k
            top_k_str = input("Number of results (default 3): ").strip()
            top_k = 3
            if top_k_str:
                try:
                    top_k = int(top_k_str)
                    is_valid, error_msg = validate_positive_int(top_k, "top_k")
                    if not is_valid:
                        print(f"Invalid input: {error_msg}")
                        continue
                except ValueError:
                    print("Please enter a valid number")
                    continue
            
            # Perform search
            results = image_search(query_path, index, store, top_k)
            
            if not results:
                print("No results found")
            else:
                print(f"\nTop {len(results)} matches:")
                for i, (score, path) in enumerate(results, 1):
                    print(f"{i}. {path} -> {float(score):.4f}")
                    
        except ValidationError as e:
            print(f"Validation error: {e}")
        except EOFError:
            break
        except Exception as e:
            logger.error(f"Search error: {e}")
            print(f"Error: {e}")


def text_search_mode(index: Index, store: Store):
    """Interactive text search mode."""
    if store.count() == 0:
        print("No images in database. Add images first.")
        return
    
    print("\n--- Text Search Mode ---")
    while True:
        try:
            query = input("Search query (or 'back' to return): ").strip()
            
            if query.lower() == "back":
                break
            
            # Validate text query
            is_valid, error_msg = validate_text_query(query)
            if not is_valid:
                print(f"Invalid query: {error_msg}")
                continue
            
            # Validate top_k
            top_k_str = input("Number of results (default 3): ").strip()
            top_k = 3
            if top_k_str:
                try:
                    top_k = int(top_k_str)
                    is_valid, error_msg = validate_positive_int(top_k, "top_k")
                    if not is_valid:
                        print(f"Invalid input: {error_msg}")
                        continue
                except ValueError:
                    print("Please enter a valid number")
                    continue
            
            # Perform search
            results = text_search(query, index, store, top_k)
            
            if not results:
                print("No results found")
            else:
                print(f"\nTop {len(results)} matches:")
                for i, (score, path) in enumerate(results, 1):
                    print(f"{i}. {path} -> {float(score):.4f}")
                    
        except ValidationError as e:
            print(f"Validation error: {e}")
        except EOFError:
            break
        except Exception as e:
            logger.error(f"Search error: {e}")
            print(f"Error: {e}")


if __name__ == "__main__":
    main()