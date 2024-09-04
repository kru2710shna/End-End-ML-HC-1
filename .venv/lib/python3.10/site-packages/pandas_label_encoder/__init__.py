from pandas import Categorical, Series
import numpy as np
from numpy import ndarray
from typing import Union, Optional, List
import pickle
import warnings


class EncoderCategoryError(Exception):
    pass


class EncoderValidationError(Exception):
    pass


class Encoder:
    '''
    Requirements
    - pandas 1.5.2 or higher
    - numpy 1.23.2 or higher

    This encoder works like scikit-learn's LabelEncoder,
    but with higher performance.
    '''
    __name__ = 'Encoder'
    __version__ = '1.0.0'
    __categories = None

    def __init__(
        self,
        x: Union[list, Series, Categorical, ndarray, None] = None,
    ):
        if x:
            self.__categories: Categorical.categories = (
                Categorical(x).categories
            )

    def fit(
        self,
        x: Union[list, Series, Categorical, ndarray]
    ) -> None:
        self.__categories = Categorical(x).categories
        return self

    @property
    def categories(self) -> Categorical.categories:
        return self.__categories

    def __check_categories(self):
        if self.__categories is None:
            raise EncoderCategoryError(
                "Make sure you fit the encoder before using its functions"
            )

    def transform(
        self,
        x: Union[list, Series, ndarray],
        validate: Optional[str] = None,
    ) -> ndarray:
        '''
        validation: 'all' or 'any'
        - 'all': if all values are in categories, return codes
        - 'any': if any values are in categories, return codes
        '''
        self.__check_categories()
        result = Categorical(x, categories=self.__categories).codes
        if validate == 'all' and (result == -1).any():
            raise EncoderValidationError('Some values are not in categories')
        if validate == 'any' and (result == -1).all():
            raise EncoderValidationError('All values are not in categories')
        return result

    def inverse_transform(
        self,
        y: Union[List[int], Series, ndarray],
        validate: Optional[str] = None
    ) -> ndarray:
        '''
        validation: 'all' or 'any'
        - 'all': if all values are in categories, return codes
        - 'any': if any values are in categories, return codes
        '''
        self.__check_categories()
        y = np.array(y)
        y_invalid_range = (y >= self.__categories.size) | (y < -1)
        y = np.where(y_invalid_range, -1, y)
        result = (
            Categorical
            .from_codes(y, categories=self.__categories)
        )
        if validate == 'all' and result.isna().any():
            raise EncoderValidationError('Some values are not in categories')
        if validate == 'any' and result.isna().all():
            raise EncoderValidationError('All values are not in categories')
        return result.to_numpy()

    def load(self, path: str):
        model: Encoder = load_encoder(path)
        self = model
        return self

    def save(self, path: str) -> None:
        save_encoder(self, path)
        return self


def load_encoder(path: str) -> Encoder:
    try:
        with open(path, 'rb') as f:
            model: Encoder = pickle.load(f)
            if model.__name__ == Encoder.__name__:
                if model.__version__ != Encoder.__version__:
                    warnings.warn(
                        'You load encoder from version '
                        + model.__version__
                        + ' but current version is '
                        + Encoder.__version__
                        + '. This may cause errors.',
                        UserWarning
                    )
                return model
            else:
                raise AttributeError
    except FileNotFoundError:
        print('File not found')
    except AttributeError:
        print('File is not an encoder')


def save_encoder(encoder: Encoder, path: str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(encoder, f)
