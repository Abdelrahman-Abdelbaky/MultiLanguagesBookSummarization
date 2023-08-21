
using System.Net;

namespace BookSummarization.Services
{
    public class FileServices : IFileServices
    {
        private readonly Microsoft.AspNetCore.Hosting.IHostingEnvironment environment;

        public FileServices(Microsoft.AspNetCore.Hosting.IHostingEnvironment environment)
        {
            this.environment = environment;
        }
        public async Task<string> UploadFile(IFormFile file)
        {
            var filePath = Path.Combine(environment.ContentRootPath, @"wwwroot/UploadedFiles",Guid.NewGuid()+file.FileName );

            using var fileStream = new FileStream(filePath, FileMode.Create);
            await file.CopyToAsync(fileStream);
            return filePath;
        }
        public void InsertToFile(List<string> text)
        {
            var fileName = /*Guid.NewGuid() + */"OutSummary.txt";
            var filePath=Path.Combine(environment.ContentRootPath, @"wwwroot/DownloadedFiles", fileName);
            using (StreamWriter outputFile = new StreamWriter(filePath))
            {
                foreach (string line in text)
                    outputFile.WriteLine(line);
                
            }
            
        }

        public MemoryStream DownloadFile(string fileName)
        {
            var path = Path.Combine(Directory.GetCurrentDirectory(), @"wwwroot/DownloadedFiles", fileName);
            var memory=new MemoryStream();
            if (File.Exists(path))
            {
                var net = new WebClient();
                var data = net.DownloadData(path);
                var content=new MemoryStream(data);
                memory = content;
            }
            memory.Position = 0;
            return memory;
        }
    }
}
