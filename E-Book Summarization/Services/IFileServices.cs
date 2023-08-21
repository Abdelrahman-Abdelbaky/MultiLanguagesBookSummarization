namespace BookSummarization.Services
{
    public interface IFileServices
    {
        public Task<string> UploadFile(IFormFile file);
        public void InsertToFile(List<string> text);
        public MemoryStream DownloadFile(string fileName);
    }
}
